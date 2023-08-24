#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 normal;

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projcam;

	vec4 cameraPos;
	vec4 lightPos[3];
	vec4 lightColor[3];
} uScene;

layout (location = 0) out vec2 v2fTexCoord;
layout (location = 1) out vec3 oNormal;
layout (location = 2) out vec3 oPosition;
layout (location = 3) out vec3 oCameraPos;
layout (location = 4) out vec3 oLightPos[3];
layout (location = 7) out vec3 oLightColor[3];
void main()
{
	v2fTexCoord = texCoord;
	oNormal = normal;
	oPosition = position;
	oCameraPos = vec3(uScene.cameraPos);
	for (int i = 0; i < 3; i++)
    {
        oLightPos[i] = vec3(uScene.lightPos[i]);
        oLightColor[i] = vec3(uScene.lightColor[i]);
    }
	gl_Position = uScene.projcam * vec4(position, 1.0f);
}
