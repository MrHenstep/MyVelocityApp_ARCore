package com.google.ar.core.codelab.rawdepth;

import android.media.Image;
import android.opengl.Matrix;
import android.util.Log;

import com.google.ar.core.Anchor;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.exceptions.NotYetAvailableException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

/**
 * Convert depth data from ARCore depth images to 3D pointclouds. Points are added by calling the
 * Raw Depth API, and reprojected into 3D space.
 */
public class DepthData {
    public static final int FLOATS_PER_POINT = 4; // X,Y,Z,confidence.


    // Convert ARCore depth images to 3D pointclouds.
//    public static FloatBuffer create(Frame frame, Anchor cameraPoseAnchor) {
    public static FloatBuffer create(CameraIntrinsics cameraIntrinsics, Image depthImage, Image confidenceImage, Anchor cameraPoseAnchor) {
        try {

//            // get depth and confidence images
//            Image depthImage = frame.acquireRawDepthImage16Bits();
//            Image confidenceImage = frame.acquireRawDepthConfidenceImage();

            // get camera instrinsics and extract model matrix from the pose for transformations
            // we use texture intrinsics to convert depth images into 3D points because the
            // depth image is aligned to the texture image space (16:9 for a start)
//            final CameraIntrinsics intrinsics = frame.getCamera().getTextureIntrinsics();
            float[] modelMatrix = new float[16];
            cameraPoseAnchor.getPose().toMatrix(modelMatrix, 0);

            // call function
            // i) to convert (with striding) to 3D points relative to camera,
            // ii) apply confidence filter,
            // iii) apply camera pose (extrinsics) to convert to 3D points in world space
            // iv) pack it into a float buffer (x, y, z, conf)
            final FloatBuffer points = convertRawDepthImagesTo3dPointBuffer(
                    depthImage, confidenceImage, cameraIntrinsics, modelMatrix);

            // tidy up and close images
//            depthImage.close();
//            confidenceImage.close();

            return points;
        } catch (Exception e) {
            // This normally means that depth data is not available yet.
            // This is normal, so you don't have to spam the logcat with this.
        }
        return null;
    }


    private static FloatBuffer convertRawDepthImagesTo3dPointBuffer(
            Image depth, Image confidence, CameraIntrinsics cameraTextureIntrinsics, float[] modelMatrix) {

        ShortBuffer depthBuffer = convertDepthImageToShortBuffer(depth);

        ByteBuffer confidenceBuffer = convertConfidenceImageToByteBuffer(confidence);

        final Image.Plane confidenceImagePlane = confidence.getPlanes()[0];



        // rescale the intrinsics because the depth image has a different resolution to the texture intrinsics
        // e.g. 160x90 vs 1920x1080
        final int[] intrinsicsDimensions = cameraTextureIntrinsics.getImageDimensions();
        final int depthWidth = depth.getWidth();
        final int depthHeight = depth.getHeight();
        final float fx =
                cameraTextureIntrinsics.getFocalLength()[0] * depthWidth / intrinsicsDimensions[0];
        final float fy =
                cameraTextureIntrinsics.getFocalLength()[1] * depthHeight / intrinsicsDimensions[1];
        final float cx =
                cameraTextureIntrinsics.getPrincipalPoint()[0] * depthWidth / intrinsicsDimensions[0];
        final float cy =
                cameraTextureIntrinsics.getPrincipalPoint()[1] * depthHeight / intrinsicsDimensions[1];

        // step (stride) is calculated so as to get as many points as possible while still under the limit
        final float maxNumberOfPointsToRender = 20000;
        int step = (int) Math.ceil(Math.sqrt(depthWidth * depthHeight / maxNumberOfPointsToRender));

        // Allocate the destination point buffer for a subsample of the depth image.
        FloatBuffer points = FloatBuffer.allocate(depthWidth / step * depthHeight / step * FLOATS_PER_POINT);

        // prep some storage for the points that can then be put into the buffer
        float[] pointCamera = new float[4];
        float[] pointWorld = new float[4];

        int numPointsUsed = 0;

        // loop over all points in the depth image
        for (int y = 0; y < depthHeight; y += step) {
            for (int x = 0; x < depthWidth; x += step) {

                // Depth images are tightly packed, so it's OK to not use row and pixel strides.
                int depthMillimeters = depthBuffer.get(y * depthWidth + x); // Depth image pixels are in mm.


                // ignore invalid pixels
                if (depthMillimeters == 0) {
                    continue;
                }

                // get the depth in metres
                final float depthMeters = depthMillimeters / 1000.0f;

                Log.i("DepthData", x + ", " + y + ", Depth: " + depthMeters);

                // Retrieve the confidence value for this pixel.
                final byte confidencePixelValue =
                        confidenceBuffer.get(
                                y * confidenceImagePlane.getRowStride()
                                        + x * confidenceImagePlane.getPixelStride());

                final float confidenceNormalized = ((float) (confidencePixelValue & 0xff)) / 255.0f;

                // ignore low confidence pixels
                if (confidenceNormalized < 0.5) {
                    continue;
                }

                numPointsUsed++;

                // Unproject the depth into a 3D point in camera coordinates.
                // note that this depthMeters is the Z component of the point,
                // i.e. perpendicular distance to the camera plane.
                // really we want the radial distance, but the assumption here is that the
                // angle is small, so radial ~ depth
                pointCamera[0] = depthMeters * (x - cx) / fx;
                pointCamera[1] = depthMeters * (cy - y) / fy;
                pointCamera[2] = -depthMeters;
                pointCamera[3] = 1;

                // Apply model matrix to transform point into world coordinates.
                // ARCore fixes origin and orientation of 3 coordinates when the ARCore session starts
                // then this matrix (which include the 4th row for translations) is w.r.t. to that original
                // coordinate frame.
                Matrix.multiplyMV(pointWorld, 0, modelMatrix, 0, pointCamera, 0);

                // and put 'em in the buffer
                points.put(pointWorld[0]); // X.
                points.put(pointWorld[1]); // Y.
                points.put(pointWorld[2]); // Z.
                points.put(confidenceNormalized);
            }
        }

        // AND THEN THE DEPTH RENDERER REVERSES ALL THIS TO GET BACK TO THE TEXTURE PLANE!!!

        Log.i("DepthData", "Number of points used: " + numPointsUsed);

        points.rewind();
        return points;
    }

    private static ShortBuffer convertDepthImageToShortBuffer(Image depth)    {

        final Image.Plane depthImagePlane = depth.getPlanes()[0];

        ByteBuffer depthByteBufferOriginal = depthImagePlane.getBuffer();
        ByteBuffer depthByteBuffer = ByteBuffer.allocate(depthByteBufferOriginal.capacity());

        depthByteBuffer.order(ByteOrder.LITTLE_ENDIAN);

        while (depthByteBufferOriginal.hasRemaining()) {
            depthByteBuffer.put(depthByteBufferOriginal.get());
        }
        depthByteBuffer.rewind();

        ShortBuffer depthBuffer = depthByteBuffer.asShortBuffer();

        return depthBuffer;
    }

    private static ByteBuffer convertConfidenceImageToByteBuffer(Image confidence)
    {
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
