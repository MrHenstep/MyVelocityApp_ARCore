precision mediump float;

uniform sampler2D u_Texture;
uniform float u_Alpha;
varying vec2 v_TexCoord;

void main() {
    vec4 texColor = texture2D(u_Texture, v_TexCoord);
    gl_FragColor = vec4(texColor.rgb, texColor.a * u_Alpha);
}
