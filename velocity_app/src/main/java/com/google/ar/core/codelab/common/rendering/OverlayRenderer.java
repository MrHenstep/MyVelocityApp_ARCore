package com.google.ar.core.codelab.common.rendering;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.util.Log;

import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class OverlayRenderer {
    private static final int COORDS_PER_VERTEX = 2;
    private static final int TEXCOORDS_PER_VERTEX = 2;
    private static final int FLOAT_SIZE = 4;

    private FloatBuffer quadCoords;
    private FloatBuffer quadTexCoords;

    private int quadProgram;
    private int quadPositionParam;
    private int quadTexCoordParam;
    private int alphaUniform;
    private int textureId = -1;

    private float alpha = 0.5f;

    // Fullscreen quad in NDC
    private static final float[] QUAD_COORDS_DATA = {
            -1.0f, -1.0f,
            1.0f, -1.0f,
            -1.0f,  1.0f,
            1.0f,  1.0f,
    };

    public void createOnGlThread(Context context, Bitmap bitmap) throws IOException {
        // Upload texture from bitmap
        textureId = loadTextureFromBitmap(bitmap);

        // Allocate geometry buffers
        quadCoords = createFloatBuffer(QUAD_COORDS_DATA);
        quadTexCoords = ByteBuffer.allocateDirect(QUAD_COORDS_DATA.length * FLOAT_SIZE)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        // Load shaders
        int vertexShader = ShaderUtil.loadGLShader("OverlayRenderer", context, GLES20.GL_VERTEX_SHADER, "shaders/screenquad.vert");
        int fragmentShader = ShaderUtil.loadGLShader("OverlayRenderer", context, GLES20.GL_FRAGMENT_SHADER, "shaders/overlay.frag");


        quadProgram = GLES20.glCreateProgram();
        GLES20.glAttachShader(quadProgram, vertexShader);
        GLES20.glAttachShader(quadProgram, fragmentShader);
        GLES20.glLinkProgram(quadProgram);
        GLES20.glUseProgram(quadProgram);

        quadPositionParam = GLES20.glGetAttribLocation(quadProgram, "a_Position");
        quadTexCoordParam = GLES20.glGetAttribLocation(quadProgram, "a_TexCoord");
        alphaUniform = GLES20.glGetUniformLocation(quadProgram, "u_Alpha");

        ShaderUtil.checkGLError("OverlayRenderer", "Program setup");
    }

    public void updateTexture(Bitmap bitmap) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
        GLUtils.texSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, bitmap);
    }

    public float[] draw(Frame frame) {

        float[] uvBounds = null;

        if (frame.hasDisplayGeometryChanged()) {
            frame.transformCoordinates2d(
                    Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                    quadCoords,
                    Coordinates2d.TEXTURE_NORMALIZED,
                    quadTexCoords
            );
        }

        draw();

        return uvBounds;
    }

    public float[] getUvBounds() {

        float[] uvBounds;

        float[] tex = new float[8];
        quadTexCoords.position(0);
        quadTexCoords.get(tex);
        quadTexCoords.position(0);  // Reset position if reused later

        // Compute min/max U and V
        float uMin = Float.MAX_VALUE, uMax = Float.MIN_VALUE;
        float vMin = Float.MAX_VALUE, vMax = Float.MIN_VALUE;

        for (int i = 0; i < 4; i++) {
            float u = tex[2 * i];
            float v = tex[2 * i + 1];
            uMin = Math.min(uMin, u);
            uMax = Math.max(uMax, u);
            vMin = Math.min(vMin, v);
            vMax = Math.max(vMax, v);
        }

        uvBounds = new float[] { uMin, vMin, uMax, vMax };

        return uvBounds;
    }


//    private void draw() {
//        quadCoords.position(0);
//        quadTexCoords.position(0);
//
//        GLES20.glEnable(GLES20.GL_BLEND);
//        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
//
//        GLES20.glUseProgram(quadProgram);
//        GLES20.glUniform1f(alphaUniform, alpha);
//
//        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
//        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
//
//        GLES20.glVertexAttribPointer(quadPositionParam, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, quadCoords);
//        GLES20.glVertexAttribPointer(quadTexCoordParam, TEXCOORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, quadTexCoords);
//
//        GLES20.glEnableVertexAttribArray(quadPositionParam);
//        GLES20.glEnableVertexAttribArray(quadTexCoordParam);
//
//        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
//
//        GLES20.glDisableVertexAttribArray(quadPositionParam);
//        GLES20.glDisableVertexAttribArray(quadTexCoordParam);
//        GLES20.glDisable(GLES20.GL_BLEND);
//    }

    private void draw() {
        quadCoords.position(0);
        quadTexCoords.position(0);

        // Save current OpenGL program and texture
        int[] previousProgram = new int[1];
        int[] previousTexture = new int[1];
        GLES20.glGetIntegerv(GLES20.GL_CURRENT_PROGRAM, previousProgram, 0);
        GLES20.glGetIntegerv(GLES20.GL_TEXTURE_BINDING_2D, previousTexture, 0);

        // Set up for overlay rendering
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

        GLES20.glUseProgram(quadProgram);
        GLES20.glUniform1f(alphaUniform, alpha);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);

        GLES20.glEnableVertexAttribArray(quadPositionParam);
        GLES20.glVertexAttribPointer(quadPositionParam, COORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, quadCoords);

        GLES20.glEnableVertexAttribArray(quadTexCoordParam);
        GLES20.glVertexAttribPointer(quadTexCoordParam, TEXCOORDS_PER_VERTEX, GLES20.GL_FLOAT, false, 0, quadTexCoords);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        // Clean up
        GLES20.glDisableVertexAttribArray(quadPositionParam);
        GLES20.glDisableVertexAttribArray(quadTexCoordParam);
        GLES20.glDisable(GLES20.GL_BLEND);

        // Restore previous OpenGL state
        GLES20.glUseProgram(previousProgram[0]);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, previousTexture[0]);
    }

    private int loadTextureFromBitmap(Bitmap bitmap) {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        int id = textures[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id);

        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
        return id;
    }

    private FloatBuffer createFloatBuffer(float[] array) {
        ByteBuffer bb = ByteBuffer.allocateDirect(array.length * FLOAT_SIZE);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(array);
        fb.position(0);
        return fb;
    }

    public void setAlpha(float a) {
        alpha = a;
    }
}
