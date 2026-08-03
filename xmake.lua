add_rules("mode.debug", "mode.release")
set_encodings("utf-8")
add_requires("openmp")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API", "ENABLE_CUDA_COMPAT_OPS")
    includes("xmake/nvidia.lua")
end

-- MetaX --
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()

option("use-mc")
    set_default(true)
    set_showmenu(true)
    set_description("Use the MACA/MC MetaX toolchain")
option_end()

if has_config("nv-gpu") and has_config("metax-gpu") then
    raise("NVIDIA and MetaX backends must be built separately")
end

if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API", "ENABLE_CUDA_COMPAT_OPS")
    if has_config("use-mc") then
        add_defines("ENABLE_METAX_MC_API")
    end
    includes("xmake/metax.lua")
end

-- Pre-optimization kernels, kept so the performance comparison in
-- ASSIGNMENT_REPORT.md can be regenerated rather than described. Off by default;
-- a baseline build is never what you want for real use.
option("baseline-kernels")
    set_default(false)
    set_showmenu(true)
    set_description("Build the pre-optimization argmax/rope/self_attention kernels for benchmarking")
option_end()

if has_config("baseline-kernels") then
    add_defines("LLAISYS_BASELINE_KERNELS")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end
    if has_config("metax-gpu") then
        add_deps("llaisys-device-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    add_packages("openmp", {public = true})

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_packages("openmp")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("metax-gpu") then
        add_deps("llaisys-device-metax")
        add_files("src/ops/*/metax/*.maca", {rule = "llaisys.maca"})
        local maca_root = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT") or "/opt/maca"
        add_linkdirs(path.join(maca_root, "lib"))
        add_links("runtime_cu", "mccompiler", "mcruntime", "mcblas")
        add_rpathdirs(path.join(maca_root, "lib"))
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
