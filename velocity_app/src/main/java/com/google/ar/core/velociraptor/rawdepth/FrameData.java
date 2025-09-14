package com.google.ar.core.velociraptor.rawdepth;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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

/**
 * Container for synchronized ARCore frame data including camera, depth, and confidence information.
 * 
 * <p>This class encapsulates all data extracted from a single ARCore frame, providing:
 * <ul>
 *   <li>Camera image data in YUV420 format with conversion to RGB Bitmap</li>
 *   <li>16-bit depth data in millimeters with confidence filtering</li>
 *   <li>8-bit confidence data for depth quality assessment</li>
 *   <li>Camera intrinsics and extrinsics for coordinate transformations</li>
 *   <li>Synchronized timestamps for temporal alignment</li>
 * </ul>
 * 
 * <p>Key coordinate systems handled:
 * <ul>
 *   <li><strong>Texture coordinates:</strong> Used by depth and confidence images (typically lower resolution)</li>
 *   <li><strong>Camera coordinates:</strong> Used by camera image (typically higher resolution)</li>
 *   <li><strong>World coordinates:</strong> 3D world space via camera pose transformation</li>
 * </ul>
 * 
 * <p>Data format specifications:
 * <ul>
 *   <li>Camera: YUV420 format with separate Y, U, V planes</li>
 *   <li>Depth: 16-bit unsigned integers in millimeters</li>
 *   <li>Confidence: 8-bit unsigned integers (0-255, normalized to 0.0-1.0)</li>
 *   <li>Point clouds: [x, y, depth/confidence, confidence] format</li>
 * </ul>
 * 
 * <p>Threading considerations:
 * <ul>
 *   <li>All data is extracted and buffered during construction</li>
 *   <li>ARCore Image objects are closed immediately after data extraction</li>
 *   <li>Buffers are cloned to prevent data corruption from ARCore cleanup</li>
 *   <li>Methods are thread-safe for read operations after construction</li>
 * </ul>
 * 
 * <p>Usage example:
 * <pre>{@code
 * FrameData frameData = new FrameData(frame);
 * if (frameData.isValid) {
 *     Bitmap cameraBitmap = frameData.getCameraBitmap();
 *     float[] depthPoints = frameData.getDepthPoints(0.5f);
 *     float[] transformedPoints = frameData.mapDepthPointsToCameraImage(depthPoints);
 * }
 * }</pre>
 */
public class FrameData {

    //<editor-fold desc="FrameData member variables">

    // === LOGGING AND IDENTIFICATION ===
    private static final String TAG = VelociraptorActivity.class.getSimpleName();

    // === DEVICE ORIENTATION ===
    // Gravity-based device rotation in degrees (0, 90, 180, 270)
    public int gravityRotationDeg;

    // === CAMERA IMAGE DATA (YUV420 format) ===
    // Y (luminance) plane buffer
    public ByteBuffer cameraBufferY;
    // U (chroma) plane buffer  
    public ByteBuffer cameraBufferU;
    // V (chroma) plane buffer
    public ByteBuffer cameraBufferV;

    // === CAMERA IMAGE DIMENSIONS AND STRIDES ===
    public int cameraWidth;           // Camera image width in pixels
    public int cameraHeight;          // Camera image height in pixels
    public int yRowStride;            // Y plane row stride (bytes per row)
    public int uRowStride;            // U plane row stride (bytes per row)
    public int vRowStride;            // V plane row stride (bytes per row)
    public int yPixelStride;          // Y plane pixel stride (bytes per pixel)
    public int uPixelStride;          // U plane pixel stride (bytes per pixel)
    public int vPixelStride;          // V plane pixel stride (bytes per pixel)

    // === DEPTH IMAGE DATA (16-bit depth values in mm) ===
    public ShortBuffer depthBuffer;   // 16-bit depth values in millimeters
    public int depthWidth;            // Depth image width in pixels
    public int depthHeight;           // Depth image height in pixels
    public int depthRowStride;        // Depth image row stride (bytes per row)
    public int depthPixelStride;      // Depth image pixel stride (bytes per pixel)

    // === CONFIDENCE IMAGE DATA (8-bit confidence values) ===
    public ByteBuffer confidenceBuffer;  // 8-bit confidence values (0-255)
    public int confidenceWidth;          // Confidence image width in pixels
    public int confidenceHeight;         // Confidence image height in pixels
    public int confidenceRowStride;      // Confidence image row stride (bytes per row)
    public int confidencePixelStride;    // Confidence image pixel stride (bytes per pixel)

    // === TIMESTAMP DATA (nanoseconds) ===
    public long frameTimestamp;       // ARCore frame timestamp
    public long cameraTimestamp;      // Camera image timestamp
    public long depthTimestamp;       // Depth image timestamp
    public long confidenceTimestamp;  // Confidence image timestamp
    public long baseTimestamp;        // Base timestamp for relative calculations

    // === CAMERA POSE AND CALIBRATION ===
    // 4x4 homogeneous transformation matrix (camera to world)
    public float[] extrinsicMatrixHom;

    // === TEXTURE INTRINSICS (for depth/confidence mapping) ===
    public int[] textureImageDimensions;    // Texture image dimensions [width, height]
    public float[] texturePrincipalPoint;   // Texture principal point [cx, cy]
    public float[] textureFocalLength;      // Texture focal length [fx, fy]

    // === CAMERA INTRINSICS (for camera image) ===
    public int[] cameraImageDimensions;     // Camera image dimensions [width, height]
    public float[] cameraPrincipalPoint;    // Camera principal point [cx, cy]
    public float[] cameraFocalLength;       // Camera focal length [fx, fy]

    // === VALIDATION FLAG ===
    public boolean isValid = false;         // True if frame data was successfully extracted
//</editor-fold>

    //<editor-fold desc="FrameData constructors & build">
    /**
     * Constructs FrameData from individual ARCore images and camera.
     * 
     * @param cameraImage YUV420 camera image
     * @param depthImage 16-bit depth image in millimeters
     * @param confidenceImage 8-bit confidence image
     * @param camera ARCore camera object with intrinsics and pose
     * @param inFrameTimestamp Frame timestamp in nanoseconds
     * @throws Exception if image extraction fails
     */
    public FrameData(Image cameraImage, Image depthImage, Image confidenceImage, Camera camera, long inFrameTimestamp) throws Exception {
        buildFrameData(cameraImage, depthImage, confidenceImage, camera, inFrameTimestamp);
    }

    /**
     * Constructs FrameData from ARCore Frame object.
     * Automatically acquires and processes all required images.
     * 
     * @param frame ARCore Frame containing camera, depth, and confidence data
     * @throws Exception if image acquisition or processing fails
     */
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

    /**
     * Builds FrameData by extracting and processing all image data from ARCore.
     * 
     * <p>Processing pipeline:
     * <ol>
     *   <li>Extract camera intrinsics (texture and image)</li>
     *   <li>Extract camera pose as 4x4 transformation matrix</li>
     *   <li>Clone camera YUV420 planes to prevent data corruption</li>
     *   <li>Convert depth image to ShortBuffer with little-endian byte order</li>
     *   <li>Convert confidence image to ByteBuffer with little-endian byte order</li>
     *   <li>Extract synchronized timestamps from all sources</li>
     *   <li>Mark as valid if all operations succeed</li>
     * </ol>
     * 
     * @param cameraImage YUV420 camera image
     * @param depthImage 16-bit depth image
     * @param confidenceImage 8-bit confidence image
     * @param camera ARCore camera with intrinsics and pose
     * @param inFrameTimestamp Frame timestamp in nanoseconds
     * @throws Exception if any data extraction step fails
     */
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

    /**
     * Safely closes ARCore Image objects to free resources.
     * 
     * @param cameraImage Camera image to close (may be null)
     * @param depthImage Depth image to close (may be null)
     * @param confImage Confidence image to close (may be null)
     */
    private void closeImages(Image cameraImage, Image depthImage, Image confImage) {
        if (cameraImage != null) cameraImage.close();
        if (depthImage != null)  depthImage.close();
        if (confImage != null)   confImage.close();
    }

    //</editor-fold>

    //<editor-fold desc="data format helper methods">

    /**
     * Converts texture intrinsics to flat float array for serialization.
     * Format: [width, height, cx, cy, fx, fy]
     * 
     * @return Float array containing texture intrinsics
     */
    public float[] textureIntrinsicsToFloatArray() {
        return intrinsicsToFloatArray(textureImageDimensions, texturePrincipalPoint, textureFocalLength);
    }

    /**
     * Converts camera intrinsics to flat float array for serialization.
     * Format: [width, height, cx, cy, fx, fy]
     * 
     * @return Float array containing camera intrinsics
     */
    public float[] cameraIntrinsicsToFloatArray() {
        return intrinsicsToFloatArray(cameraImageDimensions, cameraPrincipalPoint, cameraFocalLength);
    }

    /**
     * Returns the 4x4 homogeneous transformation matrix as float array.
     * Matrix represents camera-to-world transformation.
     * 
     * @return 16-element float array representing 4x4 matrix (row-major order)
     */
    public float[] extrinsicMatrixHomToFloatArray() {
        return extrinsicMatrixHom;
    }

    /**
     * Converts camera intrinsics components to a flat float array.
     * Concatenates image dimensions, principal point, and focal length in order.
     * 
     * @param imageDimensions Image dimensions [width, height] (may be null)
     * @param principalPoint Principal point [cx, cy] (may be null)
     * @param focalLength Focal length [fx, fy] (may be null)
     * @return Concatenated float array of all non-null components
     */
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


    /**
     * Clones an Image.Plane buffer to prevent data corruption.
     * Creates a new ByteBuffer with the same data and native byte order.
     * 
     * @param plane Image plane to clone
     * @return Cloned ByteBuffer with native byte order
     */
    private static ByteBuffer cloneBuffer(Image.Plane plane) {
        ByteBuffer src = plane.getBuffer();
        ByteBuffer copy = ByteBuffer.allocateDirect(src.remaining()).order(ByteOrder.nativeOrder());
        src.rewind();
        copy.put(src);
        copy.rewind();
        return copy;
    }

    /**
     * Converts 16-bit depth image to ShortBuffer with little-endian byte order.
     * Depth values are in millimeters as unsigned 16-bit integers.
     * 
     * @param depth 16-bit depth image
     * @return ShortBuffer containing depth values in mm
     */
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

    /**
     * Converts 8-bit confidence image to ByteBuffer with little-endian byte order.
     * Confidence values range from 0-255 (0=no confidence, 255=full confidence).
     * 
     * @param confidence 8-bit confidence image
     * @return ByteBuffer containing confidence values
     */
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
    /**
     * Converts YUV420 camera data to RGB Bitmap.
     * Performs YUV to NV21 conversion, then NV21 to JPEG to Bitmap.
     * 
     * @return RGB Bitmap of camera image, or null if conversion fails
     */
    public Bitmap getCameraBitmap() {

        try {

            // Allocate space for NV21 buffer (YUV420: 1.5 bytes per pixel)
            byte[] nv21 = new byte[cameraWidth * cameraHeight * 3 / 2];

            // Copy Y plane
            int pos = 0;
            for (int row = 0; row < cameraHeight; row++) {
                int yOffset = row * yRowStride;
                for (int col = 0; col < cameraWidth; col++) {
                    nv21[pos++] = cameraBufferY.get(yOffset + col * yPixelStride);
                }
            }

            // Interleave VU for NV21 format (UV planes are half resolution)
            int uvHeight = cameraHeight / 2;  // UV planes are subsampled by 2
            int uvWidth = cameraWidth / 2;    // UV planes are subsampled by 2

            for (int row = 0; row < uvHeight; row++) {
                int uvOffset = row * yRowStride;;
                for (int col = 0; col < uvWidth; col++) {
                    int uIndex = uvOffset + col * uPixelStride;
                    int vIndex = uvOffset + col * uPixelStride;

                    // V first, then U for NV21 format (opposite of NV12)
                    nv21[pos++] = cameraBufferV.get(vIndex);
                    nv21[pos++] = cameraBufferU.get(uIndex);
                }
            }

            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, cameraWidth, cameraHeight, null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, cameraWidth, cameraHeight), 100, out);  // 100% quality
            byte[] jpegData = out.toByteArray();

            return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.length);

        } catch (Exception e) {
            Log.e("DepthModel", "Failed to convert YUV to Bitmap", e);
            return null;
        }
    }

    /**
     * Maps depth points from texture coordinate system to camera image coordinates.
     * Performs coordinate transformation using camera intrinsics and scaling factors.
     * 
     * @param points Input points in texture coordinates [u, v, depth, confidence] format
     * @return Transformed points in camera coordinates [u, v, depth, confidence] format
     */
    public float[] mapDepthPointsToCameraImage(float[] points) {

        float[] transformedPoints = null;
        if (points.length == 0) return transformedPoints;

        // Scale texture intrinsics to actual depth image dimensions
        float fx_tex = textureFocalLength[0] * depthWidth / textureImageDimensions[0];
        float fy_tex = textureFocalLength[1] * depthHeight / textureImageDimensions[1];
        float cx_tex = texturePrincipalPoint[0] * depthWidth / textureImageDimensions[0];
        float cy_tex = texturePrincipalPoint[1] * depthHeight / textureImageDimensions[1];

        // Scale camera intrinsics to actual camera image dimensions
        float fx_cam = cameraFocalLength[0] * cameraWidth / cameraImageDimensions[0];
        float fy_cam = cameraFocalLength[1] * cameraHeight / cameraImageDimensions[1];
        float cx_cam = cameraPrincipalPoint[0] * cameraWidth / cameraImageDimensions[0];
        float cy_cam = cameraPrincipalPoint[1] * cameraHeight / cameraImageDimensions[1];

        int numPoints = points.length / 4;
        transformedPoints = new float[points.length];

        for (int i = 0; i < numPoints; ++i) {
            // Extract texture coordinates and depth/confidence
            float u = points[i*4];        // Texture u coordinate
            float v = points[i * 4 + 1];  // Texture v coordinate

            // Convert texture coordinates to normalized camera coordinates
            float x_tex = (u - cx_tex) / fx_tex;  // Normalized x (flip y for camera convention)
            float y_tex = (cy_tex - v) / fy_tex;  // Normalized y (flip y for camera convention)

            // Convert normalized coordinates to camera image coordinates
            float u_cam = x_tex * fx_cam + cx_cam;    // Camera u coordinate
            float v_cam = cy_cam - y_tex * fy_cam;    // Camera v coordinate (flip y back)

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

    /**
     * Extracts all confidence points from the confidence image.
     * Returns points in [u, v, confidence, 0] format for visualization.
     * 
     * @return Float array of confidence points, or null if extraction fails
     */
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

    /**
     * Extracts depth points with confidence filtering and subsampling.
     * Returns points in [u, v, depth, confidence] format where confidence >= threshold.
     * Uses adaptive subsampling to limit output to ~20,000 points for performance.
     * 
     * @param confidenceLimit Minimum confidence value (0.0-1.0) for point inclusion
     * @return Float array of depth points, or null if extraction fails
     */
    public float[] getDepthPoints(final float confidenceLimit) {

        float[] points = null;

        try {

            final float maxNumberOfPointsToRender = 20000;  // Performance limit for rendering
            points = new float[depthWidth * depthHeight * 4];  // 4 floats per point: [u, v, depth, confidence]
            int step = (int) Math.ceil(Math.sqrt(depthWidth * depthHeight / maxNumberOfPointsToRender));  // Subsampling step
            int numPointsUsed = 0;

            for (int iy = 0; iy < depthHeight; iy += step) {
                for (int ix = 0; ix < depthWidth; ix += step) {

                    int radialDepthMillimetres = depthBuffer.get(iy * depthWidth + ix); // Depth image pixels are in mm.

                    if (radialDepthMillimetres == 0) {  // Skip invalid depth values
                        continue;
                    }

                    float radialDepthMetres = radialDepthMillimetres / 1000.0f;  // Convert mm to meters

                    // Retrieve the confidence value for this pixel.
                    final byte confidencePixelValue =
                            confidenceBuffer.get(
                                    iy * confidenceRowStride
                                            + ix * confidencePixelStride);

                    // Normalize confidence from 0-255 to 0.0-1.0 range
                    final float confidenceNormalized = ((float) (confidencePixelValue & 0xff)) / 255.0f;
                    if (confidenceNormalized < confidenceLimit) {  // Skip low-confidence points
                        continue;
                    }

                    points[numPointsUsed * 4] = ix;
                    points[numPointsUsed * 4 + 1] = iy;
                    points[numPointsUsed * 4 + 2] = radialDepthMetres;
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
