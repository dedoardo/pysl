options('moo1', 'moo2')

@StageInput(VSin)
class VertexInpust:
    position : float3 = POSITION0
    normal : float3 = NORMAL0
    texcoord : float2 = TEXCOORD0
    color : float4 = COLOR0
    idx : uint = SV_VertexID

@StageInput(VSout)
class VertexOutput:
    position : float4 = SV_Position
    depth : float2 = NORMAL0
    normal : float3 = NORMAL1

@StageInput(PSin)
class PixelInput:
    position : float4 = SV_Position
    normal : float3 = NORMAL1
    depth : float2 = NORMAL0

@StageInput(PSout)
class PixelOutput:
    depth : float = SV_Target0
    normal : float3 = SV_Target1

class Sample:
    a : float4

@ConstantBuffer(16 * 16)
class PerObject:
    world : float4x4 = 0


@ConstantBuffer(16 * 16)
class PerFrameSep:
    view_projection : float4x4 = 0
    view : float4x4 = 16

default_sampler : register(Sampler2D, 0) = export(COW=0, MOO=1)

@VS
def VS():
    a : const.Sample
    world_pos : float4 = mul(float4(input.position, 1.0), PerObject.world)
    output.position = mul(world_pos, PerFrameSep.view_projection)
    output.normal = mul(input.normal, float3x3@aaa)
    output.normal = normalize(output.normal)
    output.depth.xy = output.position.zw

@PS
def PS():
    output.depth = input.depth.x / input.depth.y
    output.normal = input.normal
    #output.moo = default_sampler.Sample(float2(input.depth.x, 0.5), 1.0)
    #output.moo2 = default_sampler.Load(input.depth.zw, 0, 3)
    #output.moo3 = default_sampler.SampleGrad(input.normal.xx, float2(0.5, 0.5), float2(1.0, 1.0))
    #output.moo4 = default_sampler.SampleLevel(0, 0)
    #output.moo5 = default_sampler.Gather(input.normal.zz, 2, 1.0)
  #  output.normal = row4(mat, 2, 2)
    output.normal = col3(mat, 3)