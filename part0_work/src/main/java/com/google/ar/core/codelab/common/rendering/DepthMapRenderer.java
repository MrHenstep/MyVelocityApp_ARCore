package com.google.ar.core.codelab.common.rendering;

import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class DepthMapRenderer {

    private int textureId;
    private int program;
    private int positionHandle;
    private int texCoordHandle;
    private int textureHandle;

    private FloatBuffer vertexBuffer;
    private FloatBuffer texCoordBuffer;

    private static final float[] QUAD_COORDS = {
            -1.0f, -1.0f,   // Bottom left
            1.0f, -1.0f,   // Bottom right
            -1.0f,  1.0f,   // Top left
            1.0f,  1.0f    // Top right
    };

    private static final float[] QUAD_TEX_COORDS = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };

    private static final String VERTEX_SHADER =
            "attribute vec4 a_Position;" +
                    "attribute vec2 a_TexCoord;" +
                    "varying vec2 v_TexCoord;" +
                    "void main() {" +
                    "  gl_Position = a_Position;" +
                    "  v_TexCoord = a_TexCoord;" +
                    "}";

    private static final String FRAGMENT_SHADER =
            "precision mediump float;" +
                    "uniform sampler2D u_Texture;" +
                    "varying vec2 v_TexCoord;" +
                    "void main() {" +
                    "  vec4 depth = texture2D(u_Texture, v_TexCoord);" +
                    "  gl_FragColor = vec4(depth.rgb, 0.5);" +  // 0.5 = semi-transparent
                    "}";

    public void init(int viewWidth, int viewHeight) {
        float aspect = (float) viewHeight / viewWidth; // invert to stretch horizontally

        float[] quadCoords = {
                -1.0f, -1.0f,
                1.0f, -1.0f,
                -1.0f,  1.0f,
                1.0f,  1.0f
        };

        vertexBuffer = ByteBuffer.allocateDirect(quadCoords.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        vertexBuffer.put(quadCoords).position(0);

//        texCoordBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.length * 4)
//                .order(ByteOrder.nativeOrder())
//                .asFloatBuffer();
//        texCoordBuffer.put(QUAD_TEX_COORDS).position(0);

        float t0 = 0.21875f;  // bottom edge of valid image area
        float t1 = 0.78125f;  // top edge of valid image area

        float[] texCoords = {
                0.0f, t1,   // Bottom left
                1.0f, t1,   // Bottom right
                0.0f, t0,   // Top left
                1.0f, t0    // Top right
        };

        texCoordBuffer = ByteBuffer.allocateDirect(texCoords.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        texCoordBuffer.put(texCoords).position(0);


        // Compile shaders and link program
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);

        positionHandle = GLES20.glGetAttribLocation(program, "a_Position");
        texCoordHandle = GLES20.glGetAttribLocation(program, "a_TexCoord");
        textureHandle = GLES20.glGetUniformLocation(program, "u_Texture");

        // Create texture
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
    }

    public void updateBitmap(Bitmap bitmap) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
    }

    public void draw() {
        GLES20.glUseProgram(program);

        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glEnableVertexAttribArray(texCoordHandle);

        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);

        // 🔧 Disable depth testing so overlay doesn’t occlude
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);

        // 🔧 Enable alpha blending
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glUniform1i(textureHandle, 0);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        // Cleanup
        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(texCoordHandle);

        // 🔧 Restore state for later depth drawing
        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);  // <- ESSENTIAL
    }



    private int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        return shader;
    }
}
