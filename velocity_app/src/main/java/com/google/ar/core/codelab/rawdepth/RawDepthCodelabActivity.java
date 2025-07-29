/*
 * Copyright 2021 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.codelab.rawdepth;

import android.content.res.AssetFileDescriptor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;

//import android.graphics.Bitmap;
import android.media.Image;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
//import android.opengl.Matrix;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import android.view.View;

import android.widget.Toast;

import android.widget.ImageView;



import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;

import android.view.Surface;
import android.content.Context;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.codelab.common.helpers.CameraPermissionHelper;
import com.google.ar.core.codelab.common.helpers.DisplayRotationHelper;
import com.google.ar.core.codelab.common.helpers.FullScreenHelper;
import com.google.ar.core.codelab.common.helpers.SnackbarHelper;
import com.google.ar.core.codelab.common.helpers.TrackingStateHelper;

import com.google.ar.core.codelab.common.rendering.BackgroundRenderer;
import com.google.ar.core.codelab.common.rendering.DepthRenderer;
import com.google.ar.core.codelab.common.rendering.DepthMapRenderer;
import com.google.ar.core.codelab.common.rendering.OverlayRenderer;


import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import java.io.IOException;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

// Import necessary classes
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;



// If you need to convert YUV to RGB, you might need:
// import android.renderscript.Allocation;
// import android.renderscript.Element;
// import android.renderscript.RenderScript;
// import android.renderscript.ScriptIntrinsicYuvToRGB;


/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore Raw Depth API. The application will show 3D point-cloud data of the environment.
 */
public class RawDepthCodelabActivity extends AppCompatActivity implements GLSurfaceView.Renderer {
  private static final String TAG = RawDepthCodelabActivity.class.getSimpleName();

  // Rendering. The Renderers are created here, and initialized when the GL surface is created.
  private GLSurfaceView surfaceView;

  private boolean installRequested;

  private Session session;
  private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
  private DisplayRotationHelper displayRotationHelper;

  private final BackgroundRenderer backgroundRenderer = new BackgroundRenderer();
//  private final DepthRenderer depthRenderer = new DepthRenderer();

//  private final DepthMapRenderer depthMapRenderer = new DepthMapRenderer();
  private final OverlayRenderer overlayRenderer = new OverlayRenderer();
  private Bitmap latestRenderBitmap = null;

  // Inside your RawDepthCodelabActivity class:
  private Interpreter tflite_interpreter;
  private static final String MODEL_PATH = "midas_nano.tflite"; // Replace with your model filename

  private int viewWidth;
  private int viewHeight;


  // Offscreen framebuffer and texture for MiDaS letterboxed input
  private int offscreenFramebuffer = -1;
  private int offscreenTexture = -1;

  private ImageView debugDepthInputView;
  private ImageView bitmapView;

  @Override
  protected void onCreate(Bundle savedInstanceState) {

    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    bitmapView = findViewById(R.id.debug_depth_input_view);

    debugDepthInputView = findViewById(R.id.debug_depth_input_view);

    surfaceView = findViewById(R.id.surfaceview);
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    // Set up renderer.
    surfaceView.setPreserveEGLContextOnPause(true);
    surfaceView.setEGLContextClientVersion(2);
    surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
    surfaceView.setRenderer(this);
    surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    surfaceView.setWillNotDraw(false);

    installRequested = false;

    bitmapView.bringToFront();


    try {
      tflite_interpreter = new Interpreter(loadModelFile(this));
    } catch (IOException e) {
      Log.e("TFLite", "Error loading TFLite model: ", e);
    }

  }

  @Override
  protected void onResume() {
    super.onResume();

    if (session == null) {
      Exception exception = null;
      String message = null;
      try {
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }

        // ARCore requires camera permissions to operate. If we did not yet obtain runtime
        // permission on Android M and above, now is a good time to ask the user for it.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }

// Create the ARCore session.
        session = new Session(/* context= */ this);
        if (!session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)) {
          message =
                  "This device does not support the ARCore Raw Depth API. See" +
                          "https://developers.google.com/ar/devices for a list of devices that do.";
        }


      } catch (UnavailableArcoreNotInstalledException
               | UnavailableUserDeclinedInstallationException e) {
        message = "Please install ARCore";
        exception = e;
      } catch (UnavailableApkTooOldException e) {
        message = "Please update ARCore";
        exception = e;
      } catch (UnavailableSdkTooOldException e) {
        message = "Please update this app";
        exception = e;
      } catch (UnavailableDeviceNotCompatibleException e) {
        message = "This device does not support AR";
        exception = e;
      } catch (Exception e) {
        message = "Failed to create AR session";
        exception = e;
      }

      if (message != null) {
        messageSnackbarHelper.showError(this, message);
        Log.e(TAG, "Exception creating session", exception);
        return;
      }
    }

    try {
      // ************ New code to add ***************
      // Enable raw depth estimation and auto focus mode while ARCore is running.
      Config config = session.getConfig();
      config.setDepthMode(Config.DepthMode.RAW_DEPTH_ONLY);
      config.setFocusMode(Config.FocusMode.AUTO);
      session.configure(config);
      // ************ End new code to add ***************
      session.resume();
    } catch (CameraNotAvailableException e) {
      messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
      session = null;
      return;
    }

    // Note that order matters - see the note in onPause(), the reverse applies here.
    surfaceView.onResume();
    displayRotationHelper.onResume();
    messageSnackbarHelper.showMessage(this, "Waiting for depth data...");
  }

  @Override
  public void onPause() {
    super.onPause();
    if (session != null) {
      // Note that the order matters - GLSurfaceView is paused first so that it does not try
      // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
      // still call session.update() and get a SessionPausedException.
      displayRotationHelper.onPause();
      surfaceView.onPause();
      session.pause();
    }


  }

  @Override
  protected void onDestroy() {
    super.onDestroy();

    if (offscreenFramebuffer != -1) {
      int[] fb = { offscreenFramebuffer };
      GLES20.glDeleteFramebuffers(1, fb, 0);
      offscreenFramebuffer = -1;
    }

    if (offscreenTexture != -1) {
      int[] tex = { offscreenTexture };
      GLES20.glDeleteTextures(1, tex, 0);
      offscreenTexture = -1;
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
    if (!CameraPermissionHelper.hasCameraPermission(this)) {
      Toast.makeText(this, "Camera permission is needed to run this application",
              Toast.LENGTH_LONG).show();
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this);
      }
      finish();
    }
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
  }

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Prepare the rendering objects. This involves reading shaders, so may throw an IOException.
    try {
      // Create the texture and pass it to ARCore session to be filled during update().
      backgroundRenderer.createOnGlThread(/*context=*/ this);

//      depthRenderer.createOnGlThread(/*context=*/ this);

//      depthMapRenderer.init();
      // Provide a dummy placeholder bitmap on first load
      Bitmap placeholder = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
      overlayRenderer.createOnGlThread(this, placeholder);


    } catch (IOException e) {
      Log.e(TAG, "Failed to read an asset file", e);
    }
  }

  @Override
  public void onSurfaceChanged(GL10 gl, int width, int height) {
    displayRotationHelper.onSurfaceChanged(width, height);
    GLES20.glViewport(0, 0, width, height);

    viewWidth = width;
    viewHeight = height;

//    depthMapRenderer.init();  // NEW: Pass dimensions here

  }

  @Override
  public void onDrawFrame(GL10 gl) {

    // Clear screen to notify driver it should not load any pixels from previous frame.
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

    if (session == null) {
      return;
    }
    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session);

    try {

      session.setCameraTextureName(backgroundRenderer.getTextureId());

      Frame frame = session.update();
      Camera camera = frame.getCamera();

      backgroundRenderer.draw(frame);

//      depthMapRenderer.updateGeometry(frame); // updates texture coords
//      depthMapRenderer.draw();               // overlays depth

      if (latestRenderBitmap != null) {
        overlayRenderer.updateTexture(latestRenderBitmap);
      }
      overlayRenderer.draw(frame);

      runDepthEstimation(frame);

      // Retrieve the depth data for this frame.
//      FloatBuffer points = DepthData.create(frame, session.createAnchor(camera.getPose()));
//      if (points == null) {
//        return;
//      }
//
//      if (messageSnackbarHelper.isShowing() && points != null) {
//        messageSnackbarHelper.hide(this);
//      }

      // Visualize depth points.
//      depthRenderer.update(points);
//      depthRenderer.draw(camera);

      // If not tracking, show tracking failure reason instead.
//      if (camera.getTrackingState() == TrackingState.PAUSED) {
//        messageSnackbarHelper.showMessage(
//                this, TrackingStateHelper.getTrackingFailureReasonString(camera));
//        return;
//      }


    } catch (Throwable t) {
      // Avoid crashing the application due to unhandled exceptions.
      Log.e(TAG, "Exception on the OpenGL thread", t);
    }

  }

  // Helper method to load the model file
  private MappedByteBuffer loadModelFile(AppCompatActivity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  private void runDepthEstimation(Frame frame) {
    try {

      int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
      int rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);

      // using render to write to file buffer
//      Bitmap inputBitmap = renderCameraToLetterboxedSquareFBO(frame);
//      inputBitmap = flipBitmapVertically(inputBitmap);
//
//      displayBitmapWithAspectRatio(inputBitmap);

      // get camera image directly
      Image image = frame.acquireCameraImage();
      Bitmap imageBitmap = imageToBitmap(image);
      displayBitmapWithAspectRatio(imageBitmap);
      image.close();

      // centre crop
//      bitmap = cropCenterSquare(bitmap);

      // now rotate the bitmap
      Bitmap rotatedImageBitmap = rotateBitmap(imageBitmap, rotationDegrees);
      displayBitmapWithAspectRatio(rotatedImageBitmap);

      // scale to 256x256
      Bitmap scaledRotatedImageBitmap = Bitmap.createScaledBitmap(rotatedImageBitmap, 256, 256, true);
      displayBitmapWithAspectRatio(scaledRotatedImageBitmap);

       // use this to display the input bitmap as a transparent render for checking
//      inputBitmap = flipBitmapVertically(inputBitmap);
//      depthMapRenderer.updateBitmap(inputBitmap);

      Bitmap inputBitmap = scaledRotatedImageBitmap;
      Bitmap outputBitmap;

      if (true) {
        // run through Midas
        // prepare the input buffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4); // 1 image, 256x256, 3 channels, 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder());
        for (int y = 0; y < 256; y++) {
          for (int x = 0; x < 256; x++) {
            int pixel = inputBitmap.getPixel(x, y);
            inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255f); // R
            inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255f);  // G
            inputBuffer.putFloat((pixel & 0xFF) / 255f);         // B
          }
        }

        // create empty output buffer to write to
        float[][][][] outputBuffer = new float[1][256][256][1];

        // run Midas
        tflite_interpreter.run(inputBuffer, outputBuffer);

        // create coloured bitmap from Midas output buffer
        outputBitmap = createColorMappedBitmap(outputBuffer, 256, 256);
      }
      else {
        outputBitmap = scaledRotatedImageBitmap;
      }

      displayBitmapWithAspectRatio(outputBitmap);

      outputBitmap = Bitmap.createScaledBitmap(outputBitmap, 480, 640, true);
      displayBitmapWithAspectRatio(outputBitmap);

      outputBitmap = rotateBitmap(outputBitmap, -rotationDegrees);
      displayBitmapWithAspectRatio(outputBitmap);

      // flip it horizontally to match what the renderer wants
//      outputBitmap = flipBitmapHorizontally(outputBitmap);
//      displayBitmapWithAspectRatio(outputBitmap);




//      outputBitmap = flipBitmapVertically(outputBitmap);
//      outputBitmap = flipBitmapHorizontally(outputBitmap);
//      depthMapRenderer.updateBitmap(outputBitmap);

//      latestRenderBitmap = outputBitmap;
      latestRenderBitmap = imageBitmap;

      clearBitmapView();

    } catch (Exception e) {
      Log.e("Depth", "Depth estimation failed: " + e.getMessage(), e);
    }
  }


  private Bitmap imageToBitmap(Image image) {
    // IMPORTANT: Implement YUV_420_888 to RGB (Bitmap) conversion here.
    // This is a complex operation. Common ways to do it:
    // 1. RenderScript (efficient, but deprecated from API 31, works for older APIs)
    // 2. Custom Java/C++ conversion (can be slow if not optimized)
    // 3. Using a library that handles YUV conversion.

    // Placeholder - replace with actual conversion
    if (image.getFormat() != android.graphics.ImageFormat.YUV_420_888) {
      Log.e("DepthModel", "Image format is not YUV_420_888. Conversion might be incorrect.");
      // Or throw an exception
      return null;
    }

    // Example using a simple (but potentially slow for real-time) direct conversion
    // for demonstration. For production, use RenderScript or a native library.
    Image.Plane[] planes = image.getPlanes();
    ByteBuffer yBuffer = planes[0].getBuffer();
    ByteBuffer uBuffer = planes[1].getBuffer();
    ByteBuffer vBuffer = planes[2].getBuffer();

    int ySize = yBuffer.remaining();
    int uSize = uBuffer.remaining();
    int vSize = vBuffer.remaining();

    byte[] nv21 = new byte[ySize + uSize + vSize];

    yBuffer.get(nv21, 0, ySize);
    // For NV21, V comes before U. For YUV_420_888 Plane 1 is U and Plane 2 is V
    // The pixel stride for U and V planes can be 2, meaning they are interleaved.
    // This example assumes they are not interleaved and V plane follows U plane data
    // which might not be the case directly.
    // A more robust conversion is needed here.

    // This basic NV21 assembly from separate Y,U,V planes is tricky and error-prone.
    // It's highly recommended to use a library or RenderScript for YUV_420_888 to Bitmap.

    // Assuming you have a byte[] in NV21 format (which is YUV_420_SP)
    // You'd then convert NV21 to Bitmap.
    // For now, let's return a dummy bitmap if direct conversion is too complex here.
    // Log.w(TAG, "imageToBitmap: YUV_420_888 to Bitmap conversion is complex and not fully implemented here. Using a placeholder.");
    // return Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888); // Placeholder

    // A more correct (but potentially still needs optimization) way using YuvImage
    // This path assumes the image is YUV_420_888, which can be converted to NV21
    // then to JPEG, then to Bitmap. This is indirect and not the most performant.
    try {
      Image.Plane Y = planes[0];
      Image.Plane U = planes[1];
      Image.Plane V = planes[2];

      int Yb = Y.getBuffer().remaining();
      int Ub = U.getBuffer().remaining();
      int Vb = V.getBuffer().remaining();

      byte[] data = new byte[Yb + Ub + Vb];

      Y.getBuffer().get(data, 0, Yb);
      // NV21 requires VUVUVU... but U and V planes can be separate or interleaved
      // For YUV_420_888, U and V planes have pixelStride. If pixelStride is 1, they are separate.
      // If pixelStride is 2, U and V components are interleaved (e.g., UVUV...).
      // ARCore typically provides non-interleaved U and V planes (pixelStride = 1 for U, V)
      // but their row strides might differ.
      // The order in NV21 is YYYY... VUVU...

      // This is a simplified approach, direct conversion for performance is better
      ByteBuffer yuvBytes = ByteBuffer.allocateDirect(image.getWidth() * image.getHeight() * 3 / 2);

      // Copy Y plane
      yBuffer.rewind();
      yuvBytes.put(yBuffer);

      // Copy V and U planes (NV21 format: Y plane, then interleaved VU plane)
      vBuffer.rewind();
      uBuffer.rewind();

      byte[] vData = new byte[vBuffer.remaining()];
      vBuffer.get(vData);

      byte[] uData = new byte[uBuffer.remaining()];
      uBuffer.get(uData);

      // Interleave V and U data: VUVUVU...
      // Note: ARCore's YUV_420_888 provides U plane first, then V plane.
      // And pixel strides are usually 1 for U and V planes.
      // Row strides might be different from width.
      for (int row = 0; row < image.getHeight() / 2; row++) {
        for (int col = 0; col < image.getWidth() / 2; col++) {
          int vIndex = row * planes[2].getRowStride() + col * planes[2].getPixelStride();
          int uIndex = row * planes[1].getRowStride() + col * planes[1].getPixelStride();
          yuvBytes.put(vData[vIndex]);
          yuvBytes.put(uData[uIndex]);
        }
      }


      android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(
              yuvBytes.array(),
              android.graphics.ImageFormat.NV21, // YUV_420_888 can be converted to NV21
              image.getWidth(),
              image.getHeight(),
              null);
      java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
      yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);
      byte[] imageBytes = out.toByteArray();
      return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

    } catch (Exception e) {
      Log.e("DepthModel", "Error converting YUV Image to Bitmap", e);
      return null; // Or a fallback bitmap
    }
  }

  private int getCameraImageRotationDegrees(Context context, int displayRotation) {
    CameraManager cameraManager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);

    try {
      for (String cameraId : cameraManager.getCameraIdList()) {
        CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);

        Integer lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (lensFacing != null && lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
          Integer sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);

          if (sensorOrientation == null) return 0;

          // Map display rotation to degrees
          int deviceRotationDegrees;
          switch (displayRotation) {
            case Surface.ROTATION_0:
              deviceRotationDegrees = 0;
              break;
            case Surface.ROTATION_90:
              deviceRotationDegrees = 90;
              break;
            case Surface.ROTATION_180:
              deviceRotationDegrees = 180;
              break;
            case Surface.ROTATION_270:
              deviceRotationDegrees = 270;
              break;
            default:
              deviceRotationDegrees = 0;
          }

          // Calculate relative rotation
          int rotation = (sensorOrientation - deviceRotationDegrees + 360) % 360;
          return rotation;
        }
      }
    } catch (CameraAccessException e) {
      Log.e("CameraRotation", "Failed to access camera characteristics", e);
    }

    return 0;
  }

  private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
    Matrix matrix = new Matrix();
    matrix.postRotate(rotationDegrees);
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
  }

  private static Bitmap createColorMappedBitmap(float[][][][] depth, int width, int height) {
    float min = Float.MAX_VALUE;
    float max = Float.MIN_VALUE;

    // First pass: find min/max depth
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float d = depth[0][y][x][0];
        if (d < min) min = d;
        if (d > max) max = d;
      }
    }

    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    // Second pass: map each value to a color
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float normalized = (depth[0][y][x][0] - min) / (max - min);  // ∈ [0, 1]
        int color = infernoColormap(normalized);  // Choose colormap here
        bitmap.setPixel(x, y, color);
      }
    }

    return bitmap;
  }

  private static int infernoColormap(float x) {
    x = Math.max(0f, Math.min(1f, x));  // Clamp to [0, 1]

    // These constants are approximations from the Inferno colormap
    float r = (float)Math.pow(x, 0.5);
    float g = (float)Math.pow(x, 1.5);
    float b = 1.0f - x;

    int ri = (int)(255 * r);
    int gi = (int)(255 * g);
    int bi = (int)(255 * b);

    return Color.argb(255, ri, gi, bi);
  }

  private Bitmap flipBitmapVertically(Bitmap src) {
    Matrix matrix = new Matrix();
    matrix.postScale(1f, -1f); // flip vertically
    matrix.postTranslate(0f, src.getHeight()); // move it back into view
    return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), matrix, true);
  }

  private Bitmap flipBitmapHorizontally(Bitmap src) {
    Matrix matrix = new Matrix();
    matrix.postScale(-1f, 1f); // flip horizontally
//    matrix.postTranslate(0f, src.getHeight()); // move it back into view
    matrix.postTranslate(src.getWidth(), 0f); // move it back into view
    return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), matrix, true);
  }

  private Bitmap renderCameraToLetterboxedSquareFBO(Frame frame) {
    int fboSize = 256;

    // Initialize FBO and texture if needed
    if (offscreenFramebuffer == -1) {
      int[] fb = new int[1];
      GLES20.glGenFramebuffers(1, fb, 0);
      offscreenFramebuffer = fb[0];

      int[] tex = new int[1];
      GLES20.glGenTextures(1, tex, 0);
      offscreenTexture = tex[0];

      GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, offscreenTexture);
      GLES20.glTexImage2D(
              GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
              fboSize, fboSize, 0,
              GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null
      );
      GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
      GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
      GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
      GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, offscreenFramebuffer);
      GLES20.glFramebufferTexture2D(
              GLES20.GL_FRAMEBUFFER,
              GLES20.GL_COLOR_ATTACHMENT0,
              GLES20.GL_TEXTURE_2D,
              offscreenTexture,
              0
      );

      int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
      if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
        Log.e(TAG, "Framebuffer is not complete: " + status);
      }

      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    // Save current GL state
    int[] prevFramebuffer = new int[1];
    GLES20.glGetIntegerv(GLES20.GL_FRAMEBUFFER_BINDING, prevFramebuffer, 0);
    int[] prevViewport = new int[4];
    GLES20.glGetIntegerv(GLES20.GL_VIEWPORT, prevViewport, 0);

    // Bind offscreen FBO and set letterboxed viewport
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, offscreenFramebuffer);
    int targetHeight = (int) (fboSize * 9.0f / 16.0f);
    int offsetY = (fboSize - targetHeight) / 2;
    GLES20.glViewport(0, offsetY, fboSize, targetHeight);

    // Clear only the FBO (not screen)
    GLES20.glClearColor(0f, 0f, 0f, 1f);
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

    // Draw camera into FBO only
    backgroundRenderer.draw(frame);

    // Read pixels from FBO
    ByteBuffer buffer = ByteBuffer.allocateDirect(fboSize * fboSize * 4);
    buffer.order(ByteOrder.nativeOrder());
    GLES20.glReadPixels(0, 0, fboSize, fboSize, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer);

    // Restore GL state
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, prevFramebuffer[0]);
    GLES20.glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);

    // Convert to bitmap
    Bitmap bitmap = Bitmap.createBitmap(fboSize, fboSize, Bitmap.Config.ARGB_8888);
    buffer.rewind();
    bitmap.copyPixelsFromBuffer(buffer);
    return bitmap;
  }


  public void displayBitmapWithAspectRatio(Bitmap bitmap) {
    if (bitmap == null || bitmapView == null) return;

    clearBitmapView();

    runOnUiThread(() -> {
      bitmapView.setAdjustViewBounds(true);
      bitmapView.setScaleType(ImageView.ScaleType.FIT_CENTER);
      bitmapView.setAlpha(1.0f);
      bitmapView.setVisibility(View.VISIBLE);
      bitmapView.setImageBitmap(bitmap);
      bitmapView.bringToFront(); // ensure it's above GLSurfaceView
    });
  }

  public Bitmap cropCenterSquare(Bitmap source) {
    int width = source.getWidth();
    int height = source.getHeight();
    int size = Math.min(width, height);

    int xOffset = (width - size) / 2;
    int yOffset = (height - size) / 2;

    return Bitmap.createBitmap(source, xOffset, yOffset, size, size);
  }

  public void clearBitmapView() {
    if (bitmapView == null) return;

    runOnUiThread(() -> {
      bitmapView.setImageBitmap(null);
      bitmapView.setVisibility(View.GONE);
    });
  }

}





