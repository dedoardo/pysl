options("OPT_NORMAL", "OPT_TEXCOORD", "OPT_COLOR")

@StageInput
class VertexInput:
    vertex_id : uint = SV_VertexID

@StageInput
class PixelInput:
    position : float4 = SV_Position
    texcoord : float2 = Texcoord0

@StageInput
class PixelOutput:
    color : float4 = SV_Target

@ConstantBuffer(16 * 4)
class SSAO:
    inv_projection : float4x4 = 0
    projection : float4x4 = 16
    noise_scale : float2 = 32
    sample_radius : float = 34
    num_samples : uint = 36

gbuffer0 : register(Texture2D, 0)
gbuffer1 : register(Texture2D, 1)
gbuffer_sampler : SamplerState = SamplerState(
        MinFilter = LINEAR,
        MagFilter = LINEAR
    )

noise_texture : Texture2D

@VertexShader
def vertex_shader(input : VertexInput) -> PixelInput:
    output : PixelInput
    output.position.x = float(input.vertex_id / 2) * 4.0 - 1.0
    output.position.y = float(input.vertex_id % 2) * 4.0 - 1.0
    output.position.z = 1.0
    output.position.w = 1.0

    output.texcoord.x = float(input.vertex_id / 2) * 2.0
    output.texcoord.y = 1.0 - float(input.vertex_id % 2) * 2.0

    return output

def vs_pos(texcoord : float2) -> float3:
    screen_pos : float4

    screen_pos.x = texcoord.x * 2 - 1
    screen_pos.y = (1.0 - texcoord.y) * 2 - 1
    screen_pos.z = gbuffer0.Sample(gbuffer_sampler, texcoord)
    screen_pos.w = 1.0

    vs_pos : float4 = mul(screen_pos, SSAO.inv_projection)
    return vs_pos.xyz / vs_pos.w

@PixelShader
def pixel_shader(input : PixelInput) -> PixelOutput:
    output : PixelOutput
    vspos : float3 = vs_pos(input.texcoord)
    vsnormal : float3 = gbuffer1.Sample(gbuffer_sampler, input.texcoord).xyz
    rnd_vec : float3 = noise_texture.Sample(gbuffer_sampler, input.texcoord * SSAO.noise_scale).xyz

    tangent : float3 = normalize(rnd_vec - vsnormal * dot(rnd_vec, vsnormal))
    bitangent : float3 = cross(vs_normal, tangent)
    rot : float3x3 = { tangent, bitangent, vsnormal }

    depth_bias : float = 0.05
    occlusion_factor : float = 0.0
    num_samples : uint = 64
    for i in range(0, num_samples, 1):
        sample : float3 = mul(samples[i], rot)
        sample = vspos + sample * SSAO.sample_radius

        offset : float4 = float4(sample, 1.0)
        offset = mul(offset, SSAO.projection)
        offset.xyz /= offset.w
        offset.xyz = offset.xyz * 0.5 + 0.5
        offset.y = 1.0 - offset.y

        sample_depth : float = vs_pos(offset.xy).z
        range : float = 1.0 if (abs(vspos.z - sample_depth) < SSAO.sample_radius) else 0.0
        occlusion_factor += 1.0 if sample_depth <= sample.z - depth_bias else 0.0

    occlusion : float = 1.0 - (occlusion_factor / num_samples)
    output.c = occluion.xxx
    return output