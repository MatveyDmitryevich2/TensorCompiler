#ifndef MLIR_BACKEND_INTERNAL_HPP_
#define MLIR_BACKEND_INTERNAL_HPP_

#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "mlir_backend/mlir_backend.hpp"

namespace tc::detail {

[[noreturn]] void Fail(const std::string& message);

bool IsFloatType(TensorElemType elem_type);
bool IsIntegerLikeType(TensorElemType elem_type);
std::string ElemTypeToMlir(TensorElemType elem_type);
std::string MemRefTypeToMlir(const TensorType& type);
std::string DenseLiteral(const TensorData& data);
std::string SanitizeIdentifier(std::string_view value, std::string_view prefix);

std::vector<const Value*> CollectValuesByBelong(const Graph& graph, Value::BelongTo belong);
std::vector<const Value*> CollectInternalValues(const Graph& graph);
std::vector<const Operation*> CollectOperations(const Graph& graph);

const TensorType& RequireTensorType(const Value& value);
float GetFloatAttr(const AttributeMap& attrs, const std::string& name, float default_value);
int64_t GetIntAttr(const AttributeMap& attrs, const std::string& name, int64_t default_value);
std::vector<int64_t> GetIntsAttr(const AttributeMap& attrs,
                                 const std::string& name,
                                 const std::vector<int64_t>& default_value);

class ModuleEmitter {
  public:
    ModuleEmitter(const Graph& graph, MlirEmitterOptions options);

    std::string Emit();

  private:
    const Graph& graph_;
    MlirEmitterOptions options_;
    std::ostringstream out_;
    int indent_ = 0;
    size_t unique_id_ = 0;
    std::unordered_map<std::string, std::string> value_refs_;
    std::unordered_map<std::string, std::string> global_refs_;
    std::vector<const Value*> inputs_;
    std::vector<const Value*> outputs_;
    std::vector<const Value*> initializers_;
    std::vector<const Value*> temporaries_;
    std::vector<const Operation*> operations_;

    void EmitLine(const std::string& line = {});
    std::string NewSsa(std::string_view hint);
    std::string NewSymbol(std::string_view hint);

    void ValidateGraph() const;
    std::string MemRefType(const Value& value) const;
    std::string ElemType(const Value& value) const;
    const std::vector<int64_t>& ShapeOf(const Value& value) const;
    std::string RefOf(const Value& value) const;
    static std::string JoinNames(const std::vector<const Value*>& values);

    void EmitGlobals();
    void EmitFunction();

    std::string EmitIndexConst(int64_t value);
    std::string EmitNumericConst(TensorElemType elem_type, double value);
    std::string EmitLoadRaw(const std::string& memref,
                            const std::string& memref_type,
                            const std::vector<std::string>& indices,
                            std::string_view hint);
    void EmitStoreRaw(const std::string& scalar,
                      const std::string& memref,
                      const std::string& memref_type,
                      const std::vector<std::string>& indices);
    std::string EmitLoadValue(const Value& value,
                              const std::vector<std::string>& indices,
                              std::string_view hint);
    void EmitStoreValue(const std::string& scalar,
                        const Value& value,
                        const std::vector<std::string>& indices);
    std::vector<std::string> BroadcastIndices(const Value& src,
                                              const Value& dst,
                                              const std::vector<std::string>& dst_indices);
    void EmitLoopNest(const std::vector<int64_t>& shape,
                      size_t dim,
                      std::vector<std::string>& indices,
                      const std::function<void(const std::vector<std::string>&)>& body);

    std::string EmitAddLike(const std::string& lhs,
                            const std::string& rhs,
                            TensorElemType elem_type,
                            std::string_view hint);
    std::string EmitMulLike(const std::string& lhs,
                            const std::string& rhs,
                            TensorElemType elem_type,
                            std::string_view hint);

    void EmitElementwiseBinary(const Operation& op, bool is_add);
    void EmitRelu(const Operation& op);
    void EmitMatMul(const Operation& op);
    void EmitTranspose(const Operation& op);
    void EmitGemm(const Operation& op);
    void EmitConv(const Operation& op);
    void EmitOperation(const Operation& op);
};

} // namespace tc::detail

#endif // MLIR_BACKEND_INTERNAL_HPP_
