#include "mlir_backend_internal.hpp"

namespace tc::detail {

ModuleEmitter::ModuleEmitter(const Graph& graph, MlirEmitterOptions options)
    : graph_{graph}, options_{std::move(options)} {}

std::string ModuleEmitter::Emit() {
    inputs_ = CollectValuesByBelong(graph_, Value::BelongTo::kInput);
    outputs_ = CollectValuesByBelong(graph_, Value::BelongTo::kOutput);
    initializers_ = CollectValuesByBelong(graph_, Value::BelongTo::kInitializer);
    temporaries_ = CollectInternalValues(graph_);
    operations_ = CollectOperations(graph_);

    ValidateGraph();

    out_ << "module {\n";
    ++indent_;
    EmitGlobals();
    EmitFunction();
    --indent_;
    out_ << "}\n";
    return out_.str();
}

void ModuleEmitter::EmitLine(const std::string& line) {
    out_ << std::string(static_cast<size_t>(indent_ * 2), ' ') << line << '\n';
}

std::string ModuleEmitter::NewSsa(std::string_view hint) {
    return "%" + SanitizeIdentifier(hint, "v") + "_" + std::to_string(unique_id_++);
}

std::string ModuleEmitter::NewSymbol(std::string_view hint) {
    return "@" + SanitizeIdentifier(hint, "g") + "_" + std::to_string(unique_id_++);
}

void ModuleEmitter::ValidateGraph() const {
    auto validate_value = [](const Value* value) {
        const TensorType& type = RequireTensorType(*value);
        (void)MemRefTypeToMlir(type);
    };

    for (const Value* value : inputs_) {
        validate_value(value);
    }
    for (const Value* value : outputs_) {
        validate_value(value);
    }
    for (const Value* value : initializers_) {
        validate_value(value);
        if (!value->HasInitializerData()) {
            Fail("initializer value '" + value->Name() + "' has no payload");
        }
    }
    for (const Value* value : temporaries_) {
        validate_value(value);
    }
    if (outputs_.empty()) {
        Fail("graph has no outputs");
    }
}

std::string ModuleEmitter::MemRefType(const Value& value) const {
    return MemRefTypeToMlir(RequireTensorType(value));
}

std::string ModuleEmitter::ElemType(const Value& value) const {
    return ElemTypeToMlir(RequireTensorType(value).ElemType());
}

const std::vector<int64_t>& ModuleEmitter::ShapeOf(const Value& value) const {
    return RequireTensorType(value).Shape();
}

std::string ModuleEmitter::RefOf(const Value& value) const {
    auto it = value_refs_.find(value.Name());
    if (it == value_refs_.end()) {
        Fail("missing storage binding for value '" + value.Name() + "'");
    }
    return it->second;
}

std::string ModuleEmitter::JoinNames(const std::vector<const Value*>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += values[i]->Name();
    }
    return out;
}

void ModuleEmitter::EmitGlobals() {
    for (const Value* value : initializers_) {
        const std::string symbol = NewSymbol(value->Name());
        global_refs_.emplace(value->Name(), symbol);
        const TensorData& data = *value->InitializerData();
        EmitLine("memref.global \"private\" constant " + symbol + " : " + MemRefType(*value) + " = " + DenseLiteral(data));
    }
    if (!initializers_.empty()) {
        EmitLine();
    }
}

void ModuleEmitter::EmitFunction() {
    std::vector<std::string> args;
    for (const Value* value : inputs_) {
        const std::string arg_name = NewSsa("arg_" + value->Name());
        value_refs_[value->Name()] = arg_name;
        args.push_back(arg_name + ": " + MemRefType(*value));
    }
    for (const Value* value : outputs_) {
        const std::string arg_name = NewSsa("out_" + value->Name());
        value_refs_[value->Name()] = arg_name;
        args.push_back(arg_name + ": " + MemRefType(*value));
    }

    std::string signature;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i != 0) {
            signature += ", ";
        }
        signature += args[i];
    }

    EmitLine("func.func @" + SanitizeIdentifier(options_.entry_name, "entry") + "(" + signature + ") {");
    ++indent_;
    EmitLine("// graph inputs: " + JoinNames(inputs_));
    EmitLine("// graph outputs: " + JoinNames(outputs_));
    if (!initializers_.empty()) {
        EmitLine("// initializers: " + JoinNames(initializers_));
    }
    EmitLine();

    for (const Value* value : initializers_) {
        const std::string ssa = NewSsa("init_" + value->Name());
        value_refs_[value->Name()] = ssa;
        EmitLine(ssa + " = memref.get_global " + global_refs_.at(value->Name()) + " : " + MemRefType(*value));
    }
    if (!initializers_.empty()) {
        EmitLine();
    }

    for (const Value* value : temporaries_) {
        const std::string ssa = NewSsa("tmp_" + value->Name());
        value_refs_[value->Name()] = ssa;
        EmitLine(ssa + " = memref.alloc() : " + MemRefType(*value));
    }
    if (!temporaries_.empty()) {
        EmitLine();
    }

    for (const Operation* op : operations_) {
        EmitLine("// op: " + op->Name() + " (" + Operation::OpTypeToStr(op->Type()) + ")");
        EmitOperation(*op);
        EmitLine();
    }

    for (auto it = temporaries_.rbegin(); it != temporaries_.rend(); ++it) {
        EmitLine("memref.dealloc " + RefOf(**it) + " : " + MemRefType(**it));
    }
    if (!temporaries_.empty()) {
        EmitLine();
    }

    EmitLine("return");
    --indent_;
    EmitLine("}");
}

void ModuleEmitter::EmitOperation(const Operation& op) {
    switch (op.Type()) {
        case Operation::OpType::kAdd:
            EmitElementwiseBinary(op, true);
            return;
        case Operation::OpType::kMul:
            EmitElementwiseBinary(op, false);
            return;
        case Operation::OpType::kRelu:
            EmitRelu(op);
            return;
        case Operation::OpType::kMatMul:
            EmitMatMul(op);
            return;
        case Operation::OpType::kTranspose:
            EmitTranspose(op);
            return;
        case Operation::OpType::kGemm:
            EmitGemm(op);
            return;
        case Operation::OpType::kConv:
            EmitConv(op);
            return;
    }
    Fail("unsupported operation kind");
}

} // namespace tc::detail

namespace tc {

std::string MlirBackend::EmitModule(const Graph& graph, const MlirEmitterOptions& options) const {
    detail::ModuleEmitter emitter{graph, options};
    return emitter.Emit();
}

} // namespace tc
