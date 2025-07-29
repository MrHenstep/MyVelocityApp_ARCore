package com.google.ar.core.codelab.common.rendering;

import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class DepthMapRenderer {

    private static final int FLOAT_SIZE = 4;
    private static final int COORDS_PER_VERTEX = 2;
    private static final int TEXCOORDS_PER_VERTEX = 2;

    // Fullscreen quad in NDC
    private static final float[] QUAD_COORDS = {
            -1.0f, -1.0f,
            1.0f, -1.0f,
            -1.0f,  1.0f,
            1.0f,  1.0f,
    };

    // Manually flipped texture coordinates (horizontally)
    private static final float[] TEX_COORDS = {
            1.0f, 1.0f,  // top-left
            0.0f, 1.0f,  // top-right
            1.0f, 0.0f,  // bottom-left
            0.0f, 0.0f   // bottom-right
    };

    private FloatBuffer vertexBuffer;
    private FloatBuffer texCoordBuffer;

    private int program;
    private int textureId;
    private int positionHandle;
    private int texCoordHandle;
    private int textureHandle;

    private boolean initialized = false;

    // Shaders
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
                    "  vec4 color = texture2D(u_Texture, v_TexCoord);" +
                    "  gl_FragColor = vec4(color.rgb, 0.5);" +  // 60% opacity
                    "}";

    public void init() {
        // Vertex buffer
        vertexBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.length * FLOAT_SIZE)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        vertexBuffer.put(QUAD_COORDS).position(0);

        // Texture coordinate buffer (flipped)
        texCoordBuffer = ByteBuffer.allocateDirect(TEX_COORDS.length * FLOAT_SIZE)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        texCoordBuffer.put(TEX_COORDS).position(0);

        // Compile and link program
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);

        positionHandle = GLES20.glGetAttribLocation(program, "a_Position");
        texCoordHandle = GLES20.glGetAttribLocation(program, "a_TexCoord");
        textureHandle = GLES20.glGetUniformLocation(program, "u_Texture");

        // Create 2D texture
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
    }

    public void updateBitmap(Bitmap bitmap) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        if (!initialized) {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
            initialized = true;
        } else {
            GLUtils.texSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, bitmap);
        }
    }

    public void draw() {
        GLES20.glUseProgram(program);

        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glEnableVertexAttribArray(texCoordHandle);

        vertexBuffer.position(0);
        texCoordBuffer.position(0);

        GLES20.glVertexAttribPointer(positionHandle, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glVertexAttribPointer(texCoordHandle, TEXCOORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, texCoordBuffer);

        // Enable alpha blending for overlay
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLES20.glUniform1i(textureHandle, 0);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        // Cleanup
        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(texCoordHandle);
        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
    }

    private int loadShader(int type, String code) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, code);
        GLES20.glCompileShader(shader);
        return shader;
    }

    // No geometry transform — this is manual
    public void updateGeometry() {
        // no-op
    }
}
