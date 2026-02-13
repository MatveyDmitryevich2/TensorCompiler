set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # to generate compile_commands.json

add_library(tc-flags INTERFACE)

target_compile_features(tc-flags
    INTERFACE
        cxx_std_20
)

target_compile_options(tc-flags
    INTERFACE
        -fdiagnostics-color=always
 
        -Wno-interference-size

        -Wall
        -Wextra
        -Wpedantic
        # -Wshadow
        # -Wconversion
        # -Wsign-conversion
        -Wunused
        -Wformat=2
        -Wnon-virtual-dtor

        -fstack-protector-strong # inserts checks for buffer overflow
        # -fPIE -pie
        
        $<$<CONFIG:Debug>:
            -Og
            -g3
            -ggdb
            -fsanitize=address,leak,undefined
            -fno-omit-frame-pointer
        >

        $<$<CONFIG:Release>:
            -g
            -O2
            -march=native
            # -flto
            -DNDEBUG
        >
)

target_link_options(tc-flags
    INTERFACE
        -fdiagnostics-color=always

        -Wall
        -Wextra
        # -fPIE -pie
        -rdynamic

        $<$<CONFIG:Debug>:
            -Og
            -g3
            -ggdb
            -fsanitize=address,leak,undefined
            -fno-omit-frame-pointer
        >

        $<$<CONFIG:Release>:
            -g
            -O2
            -march=native
            # -flto
        >
)

target_compile_definitions(tc-flags INTERFACE
    $<$<CONFIG:Release>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>
    $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG>
)
