package com.google.ar.core.codelab.rawdepth;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import com.google.ar.core.Camera;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;

public class FrameData {

    //<editor-fold desc="FrameData member variables">

    private static final String TAG = RawDepthCodelabActivity.class.getSimpleName();

    public ByteBuffer cameraBufferY;
    public ByteBuffer cameraBufferU;
    public ByteBuffer cameraBufferV;

    public int cameraWidth;
    public int cameraHeight;
    public int yRowStride, uRowStride, vRowStride;
    public int yPixelStride, uPixelStride, vPixelStride;

    public ShortBuffer depthBuffer;
    public int depthWidth;
    public int depthHeight;
    public int depthRowStride;
    public int depthPixelStride;

    public ByteBuffer confidenceBuffer;
    public int confidenceWidth;
    public int confidenceHeight;
    public int confidenceRowStride;
    public int confidencePixelStride;

    public long frameTimestamp;
    public long cameraTimestamp;
    public long depthTimestamp;
    public long confidenceTimestamp;

    public long baseTimestamp;

    public float[] extrinsicMatrixHom;

    public int[] textureImageDimensions;
    public float[] texturePrincipalPoint;
    public float[] textureFocalLength;

    public int[] cameraImageDimensions;
    public float[] cameraPrincipalPoint;
    public float[] cameraFocalLength;


    public boolean isValid = false;
//</editor-fold>

    //<editor-fold desc="FrameData constructors & build">
    public FrameData(Image cameraImage, Image depthImage, Image confidenceImage, Camera camera, long inFrameTimestamp) throws Exception {
        buildFrameData(cameraImage, depthImage, confidenceImage, camera, inFrameTimestamp);
    }

    public FrameData(Frame frame) throws Exception {

        Image cameraImage = null, depthImage = null, confidenceImage = null;
        long ts = frame.getTimestamp();
        try {
            cameraImage = frame.acquireCameraImage();
            depthImage = frame.acquireRawDepthImage16Bits();
            confidenceImage = frame.acquireRawDepthConfidenceImage();
            Camera camera = frame.getCamera();
            buildFrameData(cameraImage, depthImage, confidenceImage, camera, ts);
        } catch (CameraNotAvailableException e) {
            Log.e(TAG, "Camera not available during update()", e);
            return;
        } catch (NotYetAvailableException e) {
            Log.e(TAG, "Depth points not yet available during acquire()", e);
            closeImages(cameraImage, depthImage, confidenceImage);
            return;
        } catch (Throwable t) {
            Log.e("onDraw", "FrameData build failed", t);

        } finally {
            if (confidenceImage != null) confidenceImage.close();
            if (depthImage != null) depthImage.close();
            if (cameraImage != null) cameraImage.close();
        }
    }

    private void buildFrameData(Image cameraImage, Image depthImage, Image confidenceImage, Camera camera, long inFrameTimestamp) throws Exception {

        try {

            CameraIntrinsics textureIntrinsics = camera.getTextureIntrinsics();
            textureImageDimensions = textureIntrinsics.getImageDimensions();
            texturePrincipalPoint = textureIntrinsics.getPrincipalPoint();
            textureFocalLength = textureIntrinsics.getFocalLength();

            CameraIntrinsics imageIntrinsics = camera.getImageIntrinsics();
            cameraImageDimensions = imageIntrinsics.getImageDimensions();
            cameraPrincipalPoint = imageIntrinsics.getPrincipalPoint();
            cameraFocalLength = imageIntrinsics.getFocalLength();

            Pose cameraPose = camera.getPose();
            extrinsicMatrixHom = new float[16];
            cameraPose.toMatrix(extrinsicMatrixHom, 0);

            Image.Plane[] camPlanes = cameraImage.getPlanes();

            cameraBufferY = cloneBuffer(camPlanes[0]);
            cameraBufferU = cloneBuffer(camPlanes[1]);
            cameraBufferV = cloneBuffer(camPlanes[2]);

            cameraWidth = cameraImage.getWidth();
            cameraHeight = cameraImage.getHeight();;
            yRowStride = camPlanes[0].getRowStride();
            uRowStride = camPlanes[1].getRowStride();
            vRowStride = camPlanes[2].getRowStride();
            yPixelStride = camPlanes[0].getPixelStride();
            uPixelStride = camPlanes[1].getPixelStride();
            vPixelStride = camPlanes[2].getPixelStride();;

            Image.Plane depthPlane = depthImage.getPlanes()[0];

            depthBuffer = convertDepthImageToShortBuffer(depthImage);
            depthWidth = depthImage.getWidth();
            depthHeight = depthImage.getHeight();
            depthRowStride = depthPlane.getRowStride();
            depthPixelStride = depthPlane.getPixelStride();

            Image.Plane confPlane = confidenceImage.getPlanes()[0];

            confidenceBuffer = convertConfidenceImageToByteBuffer(confidenceImage);
            confidenceWidth = confidenceImage.getWidth();
            confidenceHeight = confidenceImage.getHeight();
            confidenceRowStride = confPlane.getRowStride();
            confidencePixelStride = confPlane.getPixelStride();;

//        frameTimestamp = frame.getTimestamp();
            frameTimestamp = inFrameTimestamp;
            cameraTimestamp = cameraImage.getTimestamp();
            depthTimestamp = depthImage.getTimestamp();
            confidenceTimestamp = confidenceImage.getTimestamp();


            isValid = true;
        }
        catch (Exception e) {
            Log.e("CaptureFrame", "Error acquiring images", e);
            throw e;
        }
        finally {
            if (depthImage != null) depthImage.close();
            if (confidenceImage != null) confidenceImage.close();
            if (cameraImage != null) cameraImage.close();
        }

    }

    private void closeImages(Image cameraImage, Image depthImage, Image confImage) {
    if (cameraImage != null) cameraImage.close();
    if (depthImage != null)  depthImage.close();
    if (confImage != null)   confImage.close();
}

    //</editor-fold>

    //<editor-fold desc="data format helper methods">

    public float[] textureIntrinsicsToFloatArray() {
        return intrinsicsToFloatArray(textureImageDimensions, texturePrincipalPoint, textureFocalLength);
    }

    public float[] cameraIntrinsicsToFloatArray() {
        return intrinsicsToFloatArray(cameraImageDimensions, cameraPrincipalPoint, cameraFocalLength);
    }

    public float[] extrinsicMatrixHomToFloatArray() {
        return extrinsicMatrixHom;
    }

    private float[] intrinsicsToFloatArray(int[] imageDimensions, float[] principalPoint, float[] focalLength) {

        int total =
                (imageDimensions == null ? 0 : imageDimensions.length) +
                        (principalPoint  == null ? 0 : principalPoint.length) +
                        (focalLength     == null ? 0 : focalLength.length);

        float[] out = new float[total];
        int i = 0;

        if (imageDimensions != null) {
            for (int v : imageDimensions) out[i++] = (float) v;
        }
        if (principalPoint != null) {
            for (float v : principalPoint) out[i++] = v;
        }
        if (focalLength != null) {
            for (float v : focalLength) out[i++] = v;
        }

        return out;
    }


    private static ByteBuffer cloneBuffer(Image.Plane plane) {
        ByteBuffer src = plane.getBuffer();
        ByteBuffer copy = ByteBuffer.allocateDirect(src.remaining()).order(ByteOrder.nativeOrder());
        src.rewind();
        copy.put(src);
        copy.rewind();
        return copy;
    }

    private static ShortBuffer convertDepthImageToShortBuffer(Image depth)    {

        final Image.Plane depthImagePlane = depth.getPlanes()[0];

        ByteBuffer depthByteBufferOriginal = depthImagePlane.getBuffer();
        ByteBuffer depthByteBuffer = ByteBuffer.allocate(depthByteBufferOriginal.capacity());

        depthByteBuffer.order(ByteOrder.LITTLE_ENDIAN);

        depthByteBufferOriginal.rewind();
        while (depthByteBufferOriginal.hasRemaining()) {
            depthByteBuffer.put(depthByteBufferOriginal.get());
        }
        depthByteBuffer.rewind();

        return depthByteBuffer.asShortBuffer();
    }

    private static ByteBuffer convertConfidenceImageToByteBuffer(Image confidence)  {
        final Image.Plane confidenceImagePlane = confidence.getPlanes()[0];
        ByteBuffer confidenceBufferOriginal = confidenceImagePlane.getBuffer();
        ByteBuffer confidenceBuffer = ByteBuffer.allocate(confidenceBufferOriginal.capacity());
        confidenceBuffer.order(ByteOrder.LITTLE_ENDIAN);
        while (confidenceBufferOriginal.hasRemaining()) {
            confidenceBuffer.put(confidenceBufferOriginal.get());
        }
        confidenceBuffer.rewind();

        return confidenceBuffer;
    }
    //</editor-fold>

    //<editor-fold desc="Public interface methods">
    public Bitmap getCameraBitmap() {

        try {

            // Allocate space for NV21 buffer
            byte[] nv21 = new byte[cameraWidth * cameraHeight * 3 / 2];

            // Copy Y plane
            int pos = 0;
            for (int row = 0; row < cameraHeight; row++) {
                int yOffset = row * yRowStride;
                for (int col = 0; col < cameraWidth; col++) {
                    nv21[pos++] = cameraBufferY.get(yOffset + col * yPixelStride);
                }
            }

            // Interleave VU for NV21 format
            int uvHeight = cameraHeight / 2;
            int uvWidth = cameraWidth / 2;

            for (int row = 0; row < uvHeight; row++) {
                int uvOffset = row * yRowStride;;
                for (int col = 0; col < uvWidth; col++) {
                    int uIndex = uvOffset + col * uPixelStride;
                    int vIndex = uvOffset + col * uPixelStride;

                    // V first, then U for NV21
                    nv21[pos++] = cameraBufferV.get(vIndex);
                    nv21[pos++] = cameraBufferU.get(uIndex);
                }
            }

            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, cameraWidth, cameraHeight, null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, cameraWidth, cameraHeight), 100, out);
            byte[] jpegData = out.toByteArray();

            return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.length);

        } catch (Exception e) {
            Log.e("DepthModel", "Failed to convert YUV to Bitmap", e);
            return null;
        }
    }

    public float[] mapDepthPointsToCameraImage(float[] points) {

        float[] transformedPoints = null;
        if (points.length == 0) return transformedPoints;

        float fx_tex = textureFocalLength[0] * depthWidth / textureImageDimensions[0];
        float fy_tex = textureFocalLength[1] * depthHeight / textureImageDimensions[1];
        float cx_tex = texturePrincipalPoint[0] * depthWidth / textureImageDimensions[0];
        float cy_tex = texturePrincipalPoint[1] * depthHeight / textureImageDimensions[1];

        float fx_cam = cameraFocalLength[0] * cameraWidth / cameraImageDimensions[0];
        float fy_cam = cameraFocalLength[1] * cameraHeight / cameraImageDimensions[1];
        float cx_cam = cameraPrincipalPoint[0] * cameraWidth / cameraImageDimensions[0];
        float cy_cam = cameraPrincipalPoint[1] * cameraHeight / cameraImageDimensions[1];

        int numPoints = points.length / 4;
        transformedPoints = new float[points.length];

        for (int i = 0; i < numPoints; i++) {
            float u = points[i*4];
            float v = points[i * 4 + 1];

            float x_tex = (u - cx_tex) / fx_tex;
            float y_tex = (cy_tex - v) / fy_tex;

            float u_cam = x_tex * fx_cam + cx_cam;
            float v_cam = cy_cam - y_tex * fy_cam;

            transformedPoints[i * 4]     = u_cam;
            transformedPoints[i * 4 + 1] = v_cam;
            transformedPoints[i * 4 + 2] = points[i * 4 + 2]; // depth
            transformedPoints[i * 4 + 3] = points[i * 4 + 3]; // confidence

            if (u_cam < 0 || u_cam >= cameraWidth || v_cam < 0 || v_cam >= cameraHeight) {
                Log.i("DepthPointTransformation", "Point out of bounds: " + i + ", " + u_cam + ", " + v_cam);
            }
        }

        return transformedPoints;
    }

    public float[] getConfidencePoints() {

        float[] points = null;

        try {

//        final float maxNumberOfPointsToRender = 20000;
            points = new float[confidenceWidth * confidenceHeight * 4];
//        int step = (int) Math.ceil(Math.sqrt(confidenceWidth * confidenceHeight / maxNumberOfPointsToRender));
            int numPointsUsed = 0;

            for (int iy = 0; iy < confidenceHeight; ++iy) {
                for (int ix = 0; ix < confidenceWidth; ++ix) {

                    // Retrieve the confidence value for this pixel.
                    final byte confidencePixelValue =
                            confidenceBuffer.get(
                                    iy * confidenceRowStride
                                            + ix * confidencePixelStride);

                    final float confidenceNormalized = ((float) (confidencePixelValue & 0xff)) / 255.0f;

                    points[numPointsUsed * 4] = ix;
                    points[numPointsUsed * 4 + 1] = iy;
                    points[numPointsUsed * 4 + 2] = confidenceNormalized;
                    points[numPointsUsed * 4 + 3] = confidenceNormalized;

                    numPointsUsed++;
                }
            }

            points = Arrays.copyOf(points, numPointsUsed * 4);
        }

        catch (Throwable t) {
            // Avoid crashing the application due to unhandled exceptions.
            Log.e(TAG, "Exception on the OpenGL thread", t);
        }

        Log.i("MyDepth", "PointCloud size: " + (points != null ? points.length : 0) / 4);

        return points;
    }

    public float[] getDepthPoints(final float confidenceLimit) {

        float[] points = null;

        try {

            final float maxNumberOfPointsToRender = 20000;
            points = new float[depthWidth * depthHeight * 4];
            int step = (int) Math.ceil(Math.sqrt(depthWidth * depthHeight / maxNumberOfPointsToRender));
            int numPointsUsed = 0;

            for (int iy = 0; iy < depthHeight; iy += step) {
                for (int ix = 0; ix < depthWidth; ix += step) {

                    int depthMillimeters = depthBuffer.get(iy * depthWidth + ix); // Depth image pixels are in mm.

                    if (depthMillimeters == 0) {
                        continue;
                    }

                    float depthMeters = depthMillimeters / 1000.0f;

                    // Retrieve the confidence value for this pixel.
                    final byte confidencePixelValue =
                            confidenceBuffer.get(
                                    iy * confidenceRowStride
                                            + ix * confidencePixelStride);

                    final float confidenceNormalized = ((float) (confidencePixelValue & 0xff)) / 255.0f;
                    if (confidenceNormalized < confidenceLimit) {
                        continue;
                    }

                    points[numPointsUsed * 4] = ix;
                    points[numPointsUsed * 4 + 1] = iy;
                    points[numPointsUsed * 4 + 2] = depthMeters;
                    points[numPointsUsed * 4 + 3] = confidenceNormalized;

                    numPointsUsed++;
                }
            }

            points = Arrays.copyOf(points, numPointsUsed * 4);
        }

        catch (Throwable t) {
            // Avoid crashing the application due to unhandled exceptions.
            Log.e(TAG, "Exception on the OpenGL thread", t);
        }

        Log.i("MyDepth", "PointCloud size: " + points.length / 4);

        return points;
    }
    //</editor-fold>



}
