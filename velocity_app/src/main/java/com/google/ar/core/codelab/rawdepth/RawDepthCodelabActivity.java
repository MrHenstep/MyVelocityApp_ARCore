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

import android.graphics.ImageFormat;
import android.graphics.Matrix;

//import android.graphics.Bitmap;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
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
import com.google.ar.core.codelab.common.helpers.CameraPermissionHelper;
import com.google.ar.core.codelab.common.helpers.DisplayRotationHelper;
import com.google.ar.core.codelab.common.helpers.FullScreenHelper;
import com.google.ar.core.codelab.common.helpers.SnackbarHelper;

import com.google.ar.core.codelab.common.rendering.BackgroundRenderer;
import com.google.ar.core.codelab.common.rendering.DepthMapRenderer;
//import com.google.ar.core.codelab.common.rendering.OverlayRenderer;


import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

// Import necessary classes
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class RawDepthCodelabActivity extends AppCompatActivity implements GLSurfaceView.Renderer {

// <editor-fold desc="Member Variables">
  private static final String TAG = RawDepthCodelabActivity.class.getSimpleName();

  // Views
  private GLSurfaceView glSurfaceView;  // needs to be resumed/paused etc. because it live on GL rendering thread
  //  private ImageView debugDepthInputView;
  private ImageView bitmapView; // doesn't need resumed/paused etc. because it is passive

  // Renderers
  private final BackgroundRenderer backgroundRenderer = new BackgroundRenderer();
//  private final DepthRenderer depthRenderer = new DepthRenderer();

  private final DepthMapRenderer depthMapRenderer = new DepthMapRenderer();

//  private final OverlayRenderer overlayRenderer = new OverlayRenderer();
  private Bitmap latestRenderBitmap = null;


  // ARCore session
  private Session session;
  private boolean installRequested; // came from Codelab example

  // snackbar to say waiting for depth data
  private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();

  // what it says on the tin?
  private DisplayRotationHelper displayRotationHelper;

  // The Midas-2.0 nana model for relative depth
  private Interpreter tflite_interpreter;
  private static final String MODEL_PATH = "midas_nano.tflite"; // Replace with your model filename

  // keeping track of screen view size
  private int viewWidth;
  private int viewHeight;

  //  private float[] uvBounds = null;

  // Offscreen framebuffer and texture for MiDaS letterboxed input
  private int offscreenFramebuffer = -1;
  private int offscreenTexture = -1;
  private int offscreenWidth = -1;
  private int offscreenHeight = -1;

// </editor-fold>

// <editor-fold desc="Activity Lifecycle">

  @Override
  protected void onCreate(Bundle savedInstanceState) {

    // housekeeping
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // set up the member variable view IDs
    glSurfaceView = findViewById(R.id.surfaceview);
    bitmapView = findViewById(R.id.debug_depth_input_view);
//    debugDepthInputView = findViewById(R.id.debug_depth_input_view);

    // instantiate the member display rotation helper
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    // Set up the glSurfaceView, which will be used for the rendering
    glSurfaceView.setPreserveEGLContextOnPause(true);
    glSurfaceView.setEGLContextClientVersion(2);
    glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
    glSurfaceView.setRenderer(this);
    glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY); // tells GL thread to call this once per frame
    glSurfaceView.setWillNotDraw(false);

    // Assume we don't need to install
    installRequested = false;

    // make sure the bitmap view is on the top
    bitmapView.bringToFront();

    // (try to) load the model
    try {
      tflite_interpreter = new Interpreter(loadModelFile(this));
    } catch (IOException e) {
      Log.e("TFLite", "Error loading TFLite model: ", e);
    }

  }

  @Override
  protected void onResume() {
    super.onResume();

    // if the session is null (i.e. not started),
    if (session == null) {

      Exception exception = null;
      String message = null;

      try {

        // check to see if it is installed or needs installing
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }

        // check and if necessary request camera persmission
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }

        // Create the ARCore session.
        session = new Session(/* context= */ this);

        // if depth mode is not supported, set the local message variable to tell the user
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

      // if a message was set, show the message and log it.
      if (message != null) {
        messageSnackbarHelper.showError(this, message);
        Log.e(TAG, "Exception creating session", exception);
        return;
      }

    }

    try {
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
    glSurfaceView.onResume(); // calls the GL onDrawFrame() method (overridden in this class)
    displayRotationHelper.onResume(); // needs the the GL thread to be active as per the glSurfaceView.onResume() call above

//    messageSnackbarHelper.showMessage(this, "Waiting for depth data...");
  }

  @Override
  public void onPause() {
    super.onPause();
    if (session != null) {
      // Note that the order matters - GLSurfaceView is paused first so that it does not try
      // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
      // still call session.update() and get a SessionPausedException.
      displayRotationHelper.onPause();
      glSurfaceView.onPause();
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

// </editor-fold

  //<editor-fold desc="OpenGL calls">

  @Override
  // Called once when the rendering surface is first created.
  // A surface is a buffer-backed drawing target, into which we put whatever we want to render
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {

  // default background - dark grey, opaque
  GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

  // Prepare the rendering objects. This involves reading shaders, so may throw an IOException.
  try {
    // set up the GL renderers using the methods in those classes
    backgroundRenderer.createOnGlThread(/*context=*/ this);
//    depthRenderer.createOnGlThread(/*context=*/ this);

    // <editor-fold desc="Overlay renderer - not used">
    // Provide a dummy placeholder bitmap on first load
//      Bitmap placeholderBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
//      placeholderBitmap.eraseColor(Color.TRANSPARENT);  // Fill with transparent black (0 alpha)

//      overlayRenderer.createOnGlThread(this, placeholderBitmap);
//      latestRenderBitmap = placeholderBitmap;
    // </editor-fold>

//  } catch (IOException e) {
  } catch (Exception e) {
    Log.e(TAG, "Failed to read an asset file", e);
  }
}

  @Override
  // Called when the rendering surface changes
  // A surface is a buffer-backed drawing target, into which we put whatever we want to render
  // It might change if, say, the phone is rotated (when it will change orientation)
  public void onSurfaceChanged(GL10 gl, int width, int height) {
    displayRotationHelper.onSurfaceChanged(width, height);
    GLES20.glViewport(0, 0, width, height);

    viewWidth = width;
    viewHeight = height;

    depthMapRenderer.init(width, height);  // NEW: Pass dimensions here

  }

  // </editor-fold>

  //<editor-fold desc="OpenGL onDrawFram">

  // Called once per frame, since we set glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
  // If it doesn't finish before next frame is ready, that frame is dropped
  // i.e. everything waits for this to finish
  @Override
  public void onDrawFrame(GL10 gl) {

      // Clear the screen buffers - backgroundRenderer etc. will write to these with the results for this frame
      GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

      // return (i.e. do nothing) if the session is null
      if (session == null) {
          return;
      }

      // Notify ARCore session that the view size changed so that the perspective matrix and
      // the video background can be properly adjusted.
      // updates ARCores internal projecto matrix and rendering params based on screen rotation and view size
      displayRotationHelper.updateSessionIfNeeded(session);

      Image depthImage = null;
      Image confidenceImage = null;
      Image cameraImage = null;

      try {

          // tell ARCore where to write the camera image
          session.setCameraTextureName(backgroundRenderer.getTextureId());

          Frame frame = session.update();     // get the latest frame
          Camera camera = frame.getCamera();  //

//          backgroundRenderer.draw(frame); // write the frame picture to the colour buffer; GL thread (?) will display when it feels like it?

          depthImage = frame.acquireRawDepthImage16Bits();
          confidenceImage = frame.acquireRawDepthConfidenceImage();

//          FloatBuffer points = DepthData.create(
//                  frame.getCamera().getTextureIntrinsics(),
//                  depthImage, confidenceImage,
//                  session.createAnchor(camera.getPose()));
//          if (points == null) return;   // if we didn't get any points, return
//          if (messageSnackbarHelper.isShowing())
//              messageSnackbarHelper.hide(this); // now we've got depth data, drop the snackbar notification
//
//          depthRenderer.update(points);
//          depthRenderer.draw(camera);

          cameraImage = frame.acquireCameraImage();

          runDepthEstimation(cameraImage, depthImage, confidenceImage, true);

      }

      // <editor-fold desc="Overlay renderer - not used">
//      if (latestRenderBitmap != null) {
//        overlayRenderer.updateTexture(latestRenderBitmap);
//        overlayRenderer.draw(frame);
//      }
//      uvBounds = overlayRenderer.getUvBounds();
      // </editor-fold>

      catch (Throwable t) {
          // Avoid crashing the application due to unhandled exceptions.
          Log.e(TAG, "Exception on the OpenGL thread", t);
      }
      finally {
        if (depthImage != null) depthImage.close();
        if (confidenceImage != null) confidenceImage.close();
        if (cameraImage != null) cameraImage.close();
      }
  }

// </editor-fold>

// <editor-fold desc="Main runDepthEstimation method">
  private void runDepthEstimation(Image cameraImage, Image depthImage, Image confidenceImage, boolean runDepthModel) {

    int imageWidth = cameraImage.getWidth();
    int imageHeight = cameraImage.getHeight();

    Bitmap baseBitmap = null;
    Bitmap overlayBitmap = null;

    int rotationDegrees = 0;

    // try to get the camera image, process it as per runDepth input, and display
    try  {

      // Get bitmap from Camera image
      Bitmap imageBitmap = imageToBitmap(cameraImage);

      // rotate bitmap to match display orientation
      // and more importantly, align vertical axis in world with vertical axis in bitmap for Midas
      int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
      rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);
      Bitmap bitmap = rotateBitmap(imageBitmap, rotationDegrees);

      // shrink the bitmap to 256x256, which amounts to vertical real-world compression
      // we can think about padding or cropping to preserve real-world aspect later
      bitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);

      // if runDepth pathway selected, run the Midas model and create a coloured depth map
      // otherwise skip, so the original bitmap (still scaled to 256x256) is treated as the
      // coloured depth map would be - for testing transformations and inspection
      if (runDepthModel) {

          // convert bitmap to input buffer and create empty output buffer to write to
          ByteBuffer inputBuffer = bitmapToByteBuffer(bitmap);
          float[][][][] outputBuffer = new float[1][256][256][1];

          // run Midas
          tflite_interpreter.run(inputBuffer, outputBuffer);

          // create coloured bitmap from Midas output buffer
          bitmap = createColorMappedBitmap(outputBuffer, 256, 256);
      }

      bitmap = Bitmap.createScaledBitmap(bitmap, 480, 640, true);
//        displayBitmapWithAspectRatio(bitmap);

      baseBitmap = bitmap;

      // select the points with confidence > confidenceLimit
      float[] points4d = getPointCloud(depthImage, confidenceImage, 0.5f);

      int depthWidth = depthImage.getWidth();
      int depthHeight = depthImage.getHeight();

      float[] transformedPoints4d = mapDepthPointsToCameraImage(points4d, imageWidth, imageHeight, depthWidth, depthHeight);

      Bitmap arcoreBitmap = drawPointsOverlay(transformedPoints4d, imageWidth, imageHeight);
      arcoreBitmap = rotateBitmap(arcoreBitmap, rotationDegrees);

      overlayBitmap = arcoreBitmap;

      Bitmap combinedBitmap = baseBitmap.copy(Bitmap.Config.ARGB_8888, true);
      Canvas canvas = new Canvas(combinedBitmap);

      canvas.drawBitmap(overlayBitmap, 0, 0, null);

      displayBitmapWithAspectRatio(combinedBitmap);

      int sjdfhaks = 1;
    }
    catch (Exception e) {
      Log.e("MyDepth", "Depth estimation failed: " + e.getMessage(), e);

    }



    return;
  }
// </editor-fold>

// <editor-fold desc="Helper Methods - ARCore Depth">

  public static Bitmap drawPointsOverlay(float[] points, int bitmapWidth, int bitmapHeight) {

    // Create a transparent bitmap
    Bitmap overlayBitmap = Bitmap.createBitmap(bitmapWidth, bitmapHeight, Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(overlayBitmap);
    overlayBitmap.eraseColor(Color.TRANSPARENT);

    // Paint settings for dots
    Paint paint = new Paint();
    paint.setStyle(Paint.Style.FILL);
    paint.setAntiAlias(true);

    // Define depth range (in meters) for color mapping
    float minDepth = 0.0f;
    float maxDepth = 5.0f;

    float dotRadius = 2.0f;

    int pointCount = points.length / 4;

    float min = Float.MAX_VALUE;
    float max = Float.MIN_VALUE;
    float normMin = Float.MAX_VALUE;
    float normMax = Float.MIN_VALUE;

    for (int i = 0; i < pointCount; i++) {

      float x = points[i * 4];
      float y = points[i * 4 + 1];

      // Draw dot if in bounds
      if (x >= 0 && x < bitmapWidth && y >= 0 && y < bitmapHeight) {

        float depth = points[i * 4 + 2];
        float normalized = (depth - minDepth) / (maxDepth - minDepth);
        normalized = Math.max(0f, Math.min(1f, normalized));  // Clamp to [0,1]

        int colour = infernoColormap(1.0f - normalized);

        paint.setColor(colour);
        canvas.drawCircle(x, y, dotRadius, paint);

        min = Math.min(min, depth);
        max = Math.max(max, depth);
        normMin = Math.min(normMin, normalized);
        normMax = Math.max(normMax, normalized);

      }
    }

    Log.i("MyDepth", "Depth range: " + min + " to " + max);
    Log.i("MyDepth", "Normalized range: " + normMin + " to " + normMax);

    return overlayBitmap;
  }

  private static int rainbowColormap(float x) {
    x = Math.max(0f, Math.min(1f, x));  // Clamp to [0, 1]

    // Simple piecewise triangular functions to simulate rainbow
    float r = clamp(1.5f - Math.abs(4f * x - 1f));
    float g = clamp(1.5f - Math.abs(4f * x - 2f));
    float b = clamp(1.5f - Math.abs(4f * x - 3f));

    return Color.argb(255, (int)(r * 255), (int)(g * 255), (int)(b * 255));
  }

  private static float clamp(float val) {
    return Math.max(0f, Math.min(1f, val));
  }

  private static int turboColormap(float x) {
    x = Math.max(0f, Math.min(1f, x));  // Clamp to [0,1]

    float r = 0.13572138f + 4.61539260f * x - 42.66032258f * x * x + 132.13108234f * x * x * x
            - 152.94239396f * x * x * x * x + 59.28637943f * x * x * x * x * x;
    float g = 0.09140261f + 2.19418839f * x + 4.84296658f * x * x - 42.48820538f * x * x * x
            + 132.40959671f * x * x * x * x - 152.94239396f * x * x * x * x * x;
    float b = 0.10667330f + 13.01742974f * x - 48.20963662f * x * x + 72.13875343f * x * x * x
            - 40.07390243f * x * x * x * x + 8.07575687f * x * x * x * x * x;

    r = Math.max(0f, Math.min(1f, r));
    g = Math.max(0f, Math.min(1f, g));
    b = Math.max(0f, Math.min(1f, b));

    return Color.argb(255, (int)(r * 255), (int)(g * 255), (int)(b * 255));
  }

  public float[] mapDepthPointsToCameraImage(float[] points, int imageWidth, int imageHeight, int depthWidth, int depthHeight) {

    float[] transformedPoints = null;
    if (points.length == 0) return transformedPoints;

    float depthAspect = (float) depthHeight / (float) depthWidth;
    float b = depthAspect * (float) imageWidth;
    float c = (float) imageHeight - b;


    final int imageHeightOffset = (int) (c / 2.0f);

    int numPoints = points.length / 4;

    transformedPoints = new float[points.length];

    for (int i = 0; i < numPoints; i++) {
      float u = points[i*4] / depthWidth;
      float v = points[i * 4 + 1] / depthHeight;

      float imageX = u * imageWidth;
      float imageY = v * (imageHeight - 2 * imageHeightOffset) + imageHeightOffset;

      transformedPoints[i * 4]     = imageX;
      transformedPoints[i * 4 + 1] = imageY;
      transformedPoints[i * 4 + 2] = points[i * 4 + 2]; // depth
      transformedPoints[i * 4 + 3] = points[i * 4 + 3]; // confidence

      if (imageX < 0 || imageX >= imageWidth || imageY < 0 || imageY >= imageHeight) {
        Log.i("MyDepthBounds", "Point out of bounds: " + i + ", " + imageX + ", " + imageY);
      }
    }

    return transformedPoints;
  }

  private float[] getPointCloud(Image depthImage, Image confidenceImage, final float confidenceLimit) {

    float[] points = null;

    try {

      final int depthWidth = depthImage.getWidth();
      final int depthHeight = depthImage.getHeight();

      final float maxNumberOfPointsToRender = 20000;

      points = new float[depthWidth * depthHeight * 4];

      int step = (int) Math.ceil(Math.sqrt(depthWidth * depthHeight / maxNumberOfPointsToRender));

      int numPointsUsed = 0;

      ShortBuffer depthBuffer = convertDepthImageToShortBuffer(depthImage);
      ByteBuffer confidenceBuffer = convertConfidenceImageToByteBuffer(confidenceImage);

      final Image.Plane confidenceImagePlane = confidenceImage.getPlanes()[0];
      int rowStride = confidenceImagePlane.getRowStride();
      int pixelStride = confidenceImagePlane.getPixelStride();

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
                          iy * rowStride
                                  + ix * pixelStride);

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
// </editor-fold>

//<editor-fold desc="Helper Methods - buffers">

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

// <editor-fold desc="Helper Methods - images">

  private ByteBuffer bitmapToByteBuffer(Bitmap bitmap) {

    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4); // 1 image, 256x256, 3 channels, 4 bytes per float
    inputBuffer.order(ByteOrder.nativeOrder());
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        int pixel = bitmap.getPixel(x, y);
        inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255f); // R
        inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255f);  // G
        inputBuffer.putFloat((pixel & 0xFF) / 255f);         // B
      }
    }

    return inputBuffer;
  }

  private Bitmap imageToBitmap(Image image) {
    if (image.getFormat() != ImageFormat.YUV_420_888) {
      Log.e("DepthModel", "Unexpected image format: " + image.getFormat());
      return null;
    }

    try {
      int width = image.getWidth();
      int height = image.getHeight();
      Image.Plane[] planes = image.getPlanes();

      // Allocate space for NV21 buffer
      byte[] nv21 = new byte[width * height * 3 / 2];

      // Copy Y plane
      ByteBuffer yBuffer = planes[0].getBuffer();
      int yRowStride = planes[0].getRowStride();
      int yPixelStride = planes[0].getPixelStride();

      int pos = 0;
      for (int row = 0; row < height; row++) {
        int yOffset = row * yRowStride;
        for (int col = 0; col < width; col++) {
          nv21[pos++] = yBuffer.get(yOffset + col * yPixelStride);
        }
      }

      // Interleave VU for NV21 format
      ByteBuffer uBuffer = planes[1].getBuffer();
      ByteBuffer vBuffer = planes[2].getBuffer();
      int chromaRowStride = planes[1].getRowStride();
      int chromaPixelStride = planes[1].getPixelStride();

      int uvHeight = height / 2;
      int uvWidth = width / 2;

      for (int row = 0; row < uvHeight; row++) {
        int uvOffset = row * chromaRowStride;
        for (int col = 0; col < uvWidth; col++) {
          int uIndex = uvOffset + col * chromaPixelStride;
          int vIndex = uvOffset + col * chromaPixelStride;

          // V first, then U for NV21
          nv21[pos++] = vBuffer.get(vIndex);
          nv21[pos++] = uBuffer.get(uIndex);
        }
      }

      YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
      byte[] jpegData = out.toByteArray();
      return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.length);

    } catch (Exception e) {
      Log.e("DepthModel", "Failed to convert YUV to Bitmap", e);
      return null;
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

  private Bitmap getBitmapFromFrame(Frame frame) {
    int fboWidth = viewWidth;
    int fboHeight = viewHeight;

    // Initialize FBO and texture if needed
    if (offscreenFramebuffer == -1 || offscreenWidth != fboWidth || offscreenHeight != fboHeight) {
      offscreenWidth = fboWidth;
      offscreenHeight = fboHeight;

      if (offscreenFramebuffer != -1) {
        GLES20.glDeleteTextures(1, new int[]{offscreenTexture}, 0);
        GLES20.glDeleteFramebuffers(1, new int[]{offscreenFramebuffer}, 0);
      }

      int[] fb = new int[1];
      GLES20.glGenFramebuffers(1, fb, 0);
      offscreenFramebuffer = fb[0];

      int[] tex = new int[1];
      GLES20.glGenTextures(1, tex, 0);
      offscreenTexture = tex[0];

      GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, offscreenTexture);
      GLES20.glTexImage2D(
              GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
              fboWidth, fboHeight, 0,
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

    // Bind FBO and set viewport to match view size
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, offscreenFramebuffer);
    GLES20.glViewport(0, 0, fboWidth, fboHeight);

    // Clear FBO
    GLES20.glClearColor(0f, 0f, 0f, 1f);
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

    // Draw background to FBO
    backgroundRenderer.draw(frame);

    // Read pixels
    ByteBuffer buffer = ByteBuffer.allocateDirect(fboWidth * fboHeight * 4);
    buffer.order(ByteOrder.nativeOrder());
    GLES20.glReadPixels(0, 0, fboWidth, fboHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, buffer);

    // Restore previous GL state
    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, prevFramebuffer[0]);
    GLES20.glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);

    // Convert to bitmap
    Bitmap bitmap = Bitmap.createBitmap(fboWidth, fboHeight, Bitmap.Config.ARGB_8888);
    buffer.rewind();
    bitmap.copyPixelsFromBuffer(buffer);

    return bitmap;
  }

  private Bitmap getBitmapFromFrameLetterBox(Frame frame) {
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
    runOnUiThread(() -> {
      clearBitmapView();
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

  public static Bitmap cropBitmapByUV(Bitmap source, float uMin, float vMin, float uMax, float vMax) {
    if (source == null) {
      throw new IllegalArgumentException("Source bitmap is null");
    }

    // Clamp UVs to [0, 1]
    uMin = Math.max(0f, Math.min(1f, uMin));
    uMax = Math.max(0f, Math.min(1f, uMax));
    vMin = Math.max(0f, Math.min(1f, vMin));
    vMax = Math.max(0f, Math.min(1f, vMax));

    // Ensure min < max
    if (uMax <= uMin || vMax <= vMin) {
      throw new IllegalArgumentException("Invalid UV bounds");
    }

    int width = source.getWidth();
    int height = source.getHeight();

    int x = Math.round(uMin * width);
    int y = Math.round(vMin * height);
    int cropWidth = Math.round((uMax - uMin) * width);
    int cropHeight = Math.round((vMax - vMin) * height);

    // Clamp to bitmap bounds
    x = Math.max(0, Math.min(x, width - 1));
    y = Math.max(0, Math.min(y, height - 1));
    cropWidth = Math.min(cropWidth, width - x);
    cropHeight = Math.min(cropHeight, height - y);

    return Bitmap.createBitmap(source, x, y, cropWidth, cropHeight);
  }
  public void clearBitmapView() {
    if (bitmapView == null) return;

    runOnUiThread(() -> {
      bitmapView.setImageBitmap(null);
      bitmapView.setVisibility(View.GONE);
    });
  }

  public static Bitmap padBitmapToSquare(Bitmap input) {
    int width = input.getWidth();
    int height = input.getHeight();
    int size = Math.max(width, height);

    // Create a square bitmap filled with black
    Bitmap padded = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(padded);
    canvas.drawColor(Color.BLACK);

    // Compute top-left corner to center the input bitmap
    int offsetX = (size - width) / 2;
    int offsetY = (size - height) / 2;

    canvas.drawBitmap(input, offsetX, offsetY, null);
    return padded;
  }
// </editor-fold>

// <editor-fold desc="Helper methods - depth model">
  private MappedByteBuffer loadModelFile(AppCompatActivity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
// </editor-fold>

}





