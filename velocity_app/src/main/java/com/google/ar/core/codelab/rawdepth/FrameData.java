package com.google.ar.core.codelab.rawdepth;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import com.google.ar.core.Frame;

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

    //    public long frameTimestampRebased;
//    public long cameraTimestampRebased;
//    public long depthTimestampRebased;
//    public long confidenceTimestampRebased;
//
    public boolean isValid = false;
//</editor-fold>

    private void buildFrameData(Image cameraImage, Image depthImage, Image confidenceImage, long inFrameTimestamp) throws Exception {

        try {

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

    public FrameData(Image cameraImage, Image depthImage, Image confidenceImage, long inFrameTimestamp) throws Exception {
        buildFrameData(cameraImage, depthImage, confidenceImage, inFrameTimestamp);
    }

    public FrameData(Frame frame) throws Exception {

        Image cameraImage = null, depthImage = null, confidenceImage = null;
        long ts = frame.getTimestamp();
        try {
            cameraImage = frame.acquireCameraImage();
            depthImage = frame.acquireRawDepthImage16Bits();
            confidenceImage = frame.acquireRawDepthConfidenceImage();
            buildFrameData(cameraImage, depthImage, confidenceImage, ts);
        } finally {
            if (confidenceImage != null) confidenceImage.close();
            if (depthImage != null) depthImage.close();
            if (cameraImage != null) cameraImage.close();
        }
    }


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

    public Bitmap getConfidenceBitmap() {
        final int imageWidth = 640;
        final int imageHeight = 480;

        int width = confidenceWidth;   // ARCore returns 160
        int height = confidenceHeight; // and 90

        float confidenceAspect = (float) confidenceHeight / (float) confidenceWidth;
        float b = confidenceAspect * (float) imageWidth;
        float c = (float) imageHeight - b;
        final int imageHeightOffset = (int) (c / 2.0f);

        // Step 1: Create greyscale bitmap from raw confidence buffer
        Bitmap confidenceBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Bitmap targetBitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);

        for (int y = 0; y < height; y++) {

            int rowStart = y * confidenceRowStride;

            for (int x = 0; x < width; x++) {

                int imageX = x * imageWidth / confidenceWidth;
                int imageY = (y / confidenceHeight) * (imageHeight - 2 * imageHeightOffset) + imageHeightOffset;

                int offset = rowStart + x * confidencePixelStride;
                byte confidenceValue = confidenceBuffer.get(offset);
                float normalized = ((float) (confidenceValue & 0xFF)) / 255f;
                int grey = Math.round(normalized * 255);

                int argb = Color.argb(255, grey, grey, grey);

                targetBitmap.setPixel(imageX, imageY, argb);
            }
        }

        int sdfsdf=1;

        return targetBitmap;
    }


    private float[] mapDepthPointsToCameraImage(float points[]) {

        float[] transformedPoints = null;
        if (points.length == 0) return transformedPoints;

        float depthAspect = (float) depthHeight / (float) depthWidth;
        float b = depthAspect * (float) cameraWidth;
        float c = (float) cameraHeight - b;


        final int imageHeightOffset = (int) (c / 2.0f);

        int numPoints = points.length / 4;

        transformedPoints = new float[points.length];

        for (int i = 0; i < numPoints; i++) {
            float u = points[i*4] / depthWidth;
            float v = points[i * 4 + 1] / depthHeight;

            float imageX = u * cameraWidth;
            float imageY = v * (cameraHeight - 2 * imageHeightOffset) + imageHeightOffset;

            transformedPoints[i * 4]     = imageX;
            transformedPoints[i * 4 + 1] = imageY;
            transformedPoints[i * 4 + 2] = points[i * 4 + 2]; // depth
            transformedPoints[i * 4 + 3] = points[i * 4 + 3]; // confidence

            if (imageX < 0 || imageX >= cameraWidth || imageY < 0 || imageY >= cameraHeight) {
                Log.i("MyDepthBounds", "Point out of bounds: " + i + ", " + imageX + ", " + imageY);
            }
        }

        return transformedPoints;
    }

    float[] getConfidencePoints() {

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

        Log.i("MyDepth", "PointCloud size: " + points.length / 4);

        return points;
    }

    float[] getDepthPoints(final float confidenceLimit) {

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


}
