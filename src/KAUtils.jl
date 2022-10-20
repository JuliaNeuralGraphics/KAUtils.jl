module KAUtils

export linear_threads, to_device, type_from_device, device_from_type

using KernelAbstractions
using Preferences

const BACKEND = @load_preference("backend", "CPU")
@assert BACKEND ≢ nothing

linear_threads(::CPU) = Threads.nthreads()
to_device(::CPU, x) = Array(x)
type_from_device(::CPU) = Array

device_from_type(::T) where T <: Array = CPU()
device_from_type(::Type{T}) where T <: Array = CPU()
device_from_type(::Type{T}) where T <: SubArray{<:Any, <:Any, <:Array, <:Any, <:Any} = CPU()

Base.zeros(::CPU, ::Type{T}, shape) where T = zeros(T, shape)
Base.ones(::CPU, ::Type{T}, shape) where T = ones(T, shape)
Base.rand(::CPU, ::Type{T}, shape) where T = rand(T, shape)
Base.similar(::CPU, ::Type{T}, shape) where T = Array{T}(undef, shape)
Base.fill(::CPU, value, shape) = fill(value, shape)

@static if BACKEND == "ROC"
    using AMDGPU
    using ROCKernels

    linear_threads(::ROCDevice) = 512
    to_device(::ROCDevice, x) = ROCArray(x)
    type_from_device(::ROCDevice) = ROCArray

    device_from_type(::T) where T <: ROCArray = AMDGPU.default_device()
    device_from_type(::Type{T}) where T <: ROCArray = AMDGPU.default_device()
    device_from_type(::Type{T}) where T <: SubArray{<:Any, <:Any, <:ROCArray, <:Any, <:Any} = AMDGPU.default_device()

    Base.zeros(::ROCDevice, ::Type{T}, shape) where T = AMDGPU.zeros(T, shape)
    Base.ones(::ROCDevice, ::Type{T}, shape) where T = AMDGPU.ones(T, shape)
    Base.rand(::ROCDevice, ::Type{T}, shape) where T = AMDGPU.rand(T, shape)
    Base.similar(::ROCDevice, ::Type{T}, shape) where T = ROCArray{T}(undef, shape)
    Base.fill(::ROCDevice, value, shape) = AMDGPU.fill(value, shape)
elseif BACKEND == "CUDA"
    using CUDA
    using CUDAKernels

    linear_threads(::CUDADevice) = 512
    to_device(::CUDADevice, x) = CuArray(x)
    type_from_device(::CUDADevice) = CuArray

    device_from_type(::T) where T <: CuArray = CUDADevice()
    device_from_type(::Type{T}) where T <: CuArray = CUDADevice()
    device_from_type(::Type{T}) where T <: SubArray{<:Any, <:Any, <:CuArray, <:Any, <:Any} = CUDADevice()

    Base.zeros(::CUDADevice, ::Type{T}, shape) where T = CUDA.zeros(T, shape)
    Base.ones(::CUDADevice, ::Type{T}, shape) where T = CUDA.ones(T, shape)
    Base.rand(::CUDADevice, ::Type{T}, shape) where T = CUDA.rand(T, shape)
    Base.similar(::CUDADevice, ::Type{T}, shape) where T = CuArray{T}(undef, shape)
    Base.fill(::CUDADevice, value, shape) = CUDA.fill(value, shape)
end

end
