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

import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import android.view.View;

import android.widget.Button;
import android.widget.Toast;

import android.widget.ImageView;



import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;

import android.view.Surface;
import android.content.Context;

import androidx.annotation.NonNull;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Config;

import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.codelab.common.helpers.CameraPermissionHelper;
import com.google.ar.core.codelab.common.helpers.DisplayRotationHelper;
import com.google.ar.core.codelab.common.helpers.FullScreenHelper;
import com.google.ar.core.codelab.common.helpers.SnackbarHelper;

import com.google.ar.core.codelab.common.rendering.BackgroundRenderer;


import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import java.util.concurrent.atomic.AtomicBoolean;

// Import necessary classes
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RawDepthCodelabActivity extends AppCompatActivity implements GLSurfaceView.Renderer {

// <editor-fold desc="Member Variables">
  private static final String TAG = RawDepthCodelabActivity.class.getSimpleName();

  // Views
  private GLSurfaceView glSurfaceView;  // needs to be resumed/paused etc. because it live on GL rendering thread
  private ImageView bitmapView; // doesn't need resumed/paused etc. because it is passive

  // Renderers
  // *** The backgroundRenderer neatly binds the camera feed to a texture etc.
  // *** Probably we could extract this but as it works, we just turn off the drawing for now
  private final BackgroundRenderer backgroundRenderer = new BackgroundRenderer();


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

  //  private long lastProcessedTimestampNanos = 0;
  private int numFramesCollected = 0;
  private int numFramesTaken = 0;

  // Constants determining collection
  private static final int FRAME_THROTTLE_INTERVAL = 3;
  private static final int MAX_COLLECTED_FRAMES = 10;


  private Button continueButton;
  private final Object continueLock = new Object();
  private boolean continuePressed = false;

  private long baseTimestamp;

  private volatile Bitmap lastPreviewBitmap = null;

  private boolean showLiveDepthPoints = true; // draw depth points even when not collecting


  // Cached collection of frames
  private final List<FrameData> collectedFrames = Collections.synchronizedList(new ArrayList<>());

  private volatile boolean collectingFrames = false;

  private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();

  private final AtomicBoolean processingFrames = new AtomicBoolean(false);

  private int batchCounter = 0;  // starts at 0


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

//    continueButton = findViewById(R.id.continue_button);
//    continueButton.setVisibility(View.GONE);  // Hide initially

    continueButton = findViewById(R.id.continue_button);
    continueButton.setVisibility(View.VISIBLE);  // Show the button on app start

    continueButton.setOnClickListener(v -> {
      if (collectingFrames || processingFrames.get()) {
        Log.i(TAG, "Continue pressed while busy; ignoring.");
        return;
      }
      collectingFrames = true;

      // Increment batch counter here
      int thisBatch = batchCounter++;
      Log.i(TAG, "User pressed continue – starting batch " + thisBatch);

      continueButton.setText("Collecting ... (Batch " + thisBatch + ")");
      continueButton.setEnabled(false);
      continueButton.setVisibility(View.VISIBLE);
      continueButton.bringToFront();
      continueButton.setElevation(getResources().getDisplayMetrics().density * 16f);
    });




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
    analysisExecutor.shutdown();

  }

  @Override
  public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] results) {
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

//<editor-fold desc="GL thread calls">

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
  }

  // </editor-fold>

//<editor-fold desc="GL onDrawFrame">
  @Override
  public void onDrawFrame(GL10 gl) {

    if (session == null) return;

    displayRotationHelper.updateSessionIfNeeded(session);
    session.setCameraTextureName(backgroundRenderer.getTextureId());

    // keep pointers to the images and frame so we can wrap the acquires in a try-catch
    Frame frame = null;
    Image cameraImage = null;
    Image depthImage = null;
    Image confImage = null;

    // try to acquire the frame and images
    try {
      frame = session.update();
      cameraImage = frame.acquireCameraImage();
      depthImage = frame.acquireRawDepthImage16Bits();
      confImage = frame.acquireRawDepthConfidenceImage();
    } catch (CameraNotAvailableException e) {
      Log.e(TAG, "Camera not available during update()", e);
      if (cameraImage != null) cameraImage.close();
      if (depthImage != null) depthImage.close();
      if (confImage != null) confImage.close();
      return;
    } catch (NotYetAvailableException e) {
      Log.e(TAG, "Depth points not yet available during acquire()", e);
      if (cameraImage != null) cameraImage.close();
      if (depthImage != null) depthImage.close();
      if (confImage != null) confImage.close();
      return;
    }

    // check the phone rotatoin
    int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
    int rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);

    // draw preview and points using the Images we acquired
    renderCameraAndPoints(cameraImage, depthImage, confImage, rotationDegrees);

    // if we're not collecting frames, close the images safely and return
    if (!collectingFrames) {
      if (cameraImage != null) cameraImage.close();
      if (depthImage != null) depthImage.close();
      if (confImage != null) confImage.close();
      return;
    }

    // if we're throttling and rejecting this frame, close the images safely and return.
    numFramesTaken++;
    if (numFramesTaken % FRAME_THROTTLE_INTERVAL != 0) {
      if (cameraImage != null) cameraImage.close();
      if (depthImage != null)  depthImage.close();
      if (confImage != null)   confImage.close();
      return;
    }

    //
    try {

      // Create the FrameData object from the Images
      FrameData frameData = new FrameData(cameraImage, depthImage, confImage, frame.getTimestamp());

      // if the FrameData is valid,  proceed ...
      if (frameData.isValid) {

        numFramesCollected++;

        if (numFramesCollected == 1) baseTimestamp = frameData.frameTimestamp;
        frameData.baseTimestamp = baseTimestamp;

        float frameTimestampSeconds = (frameData.frameTimestamp - baseTimestamp) / 1_000_000_000f;
        float cameraTimestampSeconds = (frameData.cameraTimestamp - baseTimestamp) / 1_000_000_000f;
        float depthTimestampSeconds = (frameData.depthTimestamp - baseTimestamp) / 1_000_000_000f;
        float confidenceTimestampSeconds = (frameData.confidenceTimestamp - baseTimestamp) / 1_000_000_000f;

        // log the frame number and the timestamps
        Log.i("CollectFrames", String.format(
                "Frame, %d, Frame Timestamp, %f, Camera Timestamp, %f, Depth Timestamp, %f, Confidence Timestamp, %f",
                numFramesCollected -1, frameTimestampSeconds, cameraTimestampSeconds, depthTimestampSeconds, confidenceTimestampSeconds));


        // synchronise this section
        synchronized (collectedFrames) {

          // store the new frameData
          collectedFrames.add(frameData);

          // if we've collected the whole batch, process
          if (collectedFrames.size() >= MAX_COLLECTED_FRAMES) {

            Log.i("CollectFrames", collectedFrames.size() + " Frames Collected");

            final int thisBatch = batchCounter - 1;

            // notify the user that we're processing
            runOnUiThread(() -> {
              continueButton.setText("Processing ... (Batch " + thisBatch + ")");
              continueButton.setEnabled(false);
              continueButton.setVisibility(View.VISIBLE);
              continueButton.bringToFront();
              continueButton.setElevation(getResources().getDisplayMetrics().density * 16f); // ~16dp
            });

            collectingFrames = false;
            processingFrames.set(true);

            List<FrameData> snapshot = new ArrayList<>(collectedFrames);
            collectedFrames.clear();

            // run the number-crunching on an executor so we don't block either the GL or the UI thread
            analysisExecutor.execute(() -> {
              try {
                // loop through all the collected frames
                for (int iFrame = 0; iFrame < snapshot.size(); ++iFrame) {
                  runDepthEstimationCached(snapshot.get(iFrame), iFrame, thisBatch);
                }
              } finally {
                processingFrames.set(false);
                runOnUiThread(() -> {
                  // Ready for another round: make the button usable again.
                  continueButton.setText("Continue");
                  continueButton.setEnabled(true);
                  continueButton.setVisibility(View.VISIBLE);
                });
              }
            });
          }

        }
      }
    }

    catch (Throwable t) {
        // Avoid crashing the application due to unhandled exceptions.
        Log.e("Frame Timestamps", "Exception on the OpenGL thread", t);
    }

  }
// </editor-fold>

  private void renderCameraAndPoints(
          Image cameraImage,
          Image depthImage,
          Image confImage,
          int rotationDegrees) {

    if (cameraImage == null) {
      // Nothing new this frame; if we have a previous preview, keep showing it.
      if (lastPreviewBitmap != null) {
        Bitmap finalBmp = lastPreviewBitmap;
        runOnUiThread(() -> displayBitmapWithAspectRatio(finalBmp));
      }
      return;
    }

    // 1) Camera preview (pre-rotation width/height)
    final int camW = cameraImage.getWidth();
    final int camH = cameraImage.getHeight();

    Bitmap preview = imageToBitmap(cameraImage);
    if (preview == null) {
      if (lastPreviewBitmap != null) {
        Bitmap finalBmp = lastPreviewBitmap;
        runOnUiThread(() -> displayBitmapWithAspectRatio(finalBmp));
      }
      return;
    }

    // Rotate preview into display orientation and remember it
    preview = rotateBitmap(preview, rotationDegrees);
    lastPreviewBitmap = preview;

    // 2) Optional depth points overlay
    Bitmap composed = preview;
    if (showLiveDepthPoints && depthImage != null && confImage != null) {
      try {
        // Build points in depth space
        float[] pts = getDepthPoints(depthImage, confImage, /*confidenceLimit=*/0.5f);
        if (pts != null && pts.length >= 4) {
          int depthW = depthImage.getWidth();
          int depthH = depthImage.getHeight();

          // Map into *pre-rotation* camera space
          float[] mapped = mapDepthPointsToCameraImage(pts, camW, camH, depthW, depthH);

          // Make overlay in pre-rotation space, then rotate it to match preview
          Bitmap overlay = drawPointsOverlay(mapped, camW, camH);
          overlay = rotateBitmap(overlay, rotationDegrees);

          // Composite overlay onto preview
          composed = preview.copy(Bitmap.Config.ARGB_8888, true);
          Canvas c = new Canvas(composed);
          c.drawBitmap(overlay, 0, 0, null);
        }
      } catch (Throwable t) {
        // If anything goes wrong, just show the preview without overlay
        Log.w(TAG, "Live depth overlay failed; showing preview only.", t);
      }
    }

    // 3) Push to UI
    Bitmap finalBmp = composed;
    runOnUiThread(() -> displayBitmapWithAspectRatio(finalBmp));
  }

// <editor-fold desc="Main runDepthEstimation method">
  private void runDepthEstimationCached(FrameData frameData, int frameNumber, int batchNumber) {

  try  {

    int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
    int rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);

    Bitmap cameraBitmap = frameData.getCameraBitmap();
    cameraBitmap = rotateBitmap(cameraBitmap, rotationDegrees);
    Bitmap depthModelInputBitmap = Bitmap.createScaledBitmap(cameraBitmap, 256, 256, true);

    // get the raw depth image, extract the depth points, and transform it to camera image space
    // and then close the images
    float[] points4d = frameData.getDepthPoints(0.5f);
    float[] transformedPoints4d = null;
    if (points4d != null) transformedPoints4d = mapDepthPointsToCameraImage(points4d, frameData.cameraWidth, frameData.cameraHeight, frameData.depthWidth, frameData.depthHeight);

    float[] confidencePoints4d = frameData.getConfidencePoints();
    float[] transformedConfidencePoints4d = null;
    if (confidencePoints4d != null) transformedConfidencePoints4d = mapDepthPointsToCameraImage(confidencePoints4d, frameData.cameraWidth, frameData.cameraHeight, frameData.depthWidth, frameData.depthHeight);

    ByteBuffer inputBuffer = bitmapToByteBuffer(depthModelInputBitmap);           // convert bitmap to input buffer and create empty output buffer to write to
      float[][][][] outputBuffer = new float[1][256][256][1];
      tflite_interpreter.run(inputBuffer, outputBuffer);  // run MIDAS

      ColourMapResult colourMapResult = createColorMappedBitmap(outputBuffer, 256, 256); // create coloured bitmap from Midas output buffer

      Bitmap depthModelBitmapColour = Bitmap.createScaledBitmap(colourMapResult.colourBitmap, 480, 640, true);
      Bitmap depthModelBitmapGrey = Bitmap.createScaledBitmap(colourMapResult.greyscaleBitmap, 480, 640, true);

    // display the depth map as a transparent overlay over the camera image
    Bitmap combinedBitmap = cameraBitmap.copy(Bitmap.Config.ARGB_8888, true);
    Canvas canvas = new Canvas(combinedBitmap);

      Paint transparentPaint = new Paint();
      transparentPaint.setAlpha(128); // or tune as needed (0=transparent, 255=opaque)
      canvas.drawBitmap(depthModelBitmapColour, 0, 0, transparentPaint);

    // draw the depth points over the camera and depth map images
    if (transformedPoints4d != null) {
      Bitmap depthPointsImage = drawPointsOverlay(transformedPoints4d, frameData.cameraWidth, frameData.cameraHeight);
      depthPointsImage = rotateBitmap(depthPointsImage, rotationDegrees);
      canvas.drawBitmap(depthPointsImage, 0, 0, null);
    }

//    displayBitmapWithAspectRatio(combinedBitmap);

//    if (numFramesTaken <= 10) {


      // TIMESTAMPS
      float[] timeStamps = new float[] {
              frameData.frameTimestamp - frameData.baseTimestamp,
              frameData.cameraTimestamp - frameData.baseTimestamp,
              frameData.depthTimestamp - frameData.baseTimestamp,
              frameData.confidenceTimestamp - frameData.baseTimestamp
      };
      saveFloatArrayToBinary(timeStamps, String.format("batch_%d_timestamps_%d.bin", batchNumber, frameNumber));
      Log.i("CollectFrames", String.format("Saved timestamps for batch %d, frame %d", batchNumber, frameNumber));

      // DEPTH POINTS
      if (transformedPoints4d != null) {
        saveFloatArrayToBinary(transformedPoints4d, String.format("batch_%d_depth_points_%d.bin", batchNumber, frameNumber));
        Log.i("CollectFrames", String.format("Saved depth points for batch %d, frame %d", batchNumber, frameNumber));
      }

      // CAMERA IMAGE
      float[] cameraBitmapFloatArray = convertBitmapToFloatPointsArray(cameraBitmap);
      saveFloatArrayToBinary(cameraBitmapFloatArray, String.format("batch_%d_depth_map_camera_%d.bin", batchNumber, frameNumber));
      Log.i("CollectFrames", String.format("Saved depth map camera for batch %d, frame %d", batchNumber, frameNumber));

      // COLOUR DEPTH MAP
      float[] colourBitmapFloatArray = convertBitmapToFloatPointsArray(depthModelBitmapColour);
      saveFloatArrayToBinary(colourBitmapFloatArray, String.format("batch_%d_depth_map_colour_%d.bin", batchNumber, frameNumber));
      Log.i("CollectFrames", String.format("Saved depth map colour for batch %d, frame %d", batchNumber, frameNumber));

      // GREYSCALE DEPTH MAP based on reciprocal depth
      float[] greyBitmapFloatArray = convertBitmapToFloatPointsArray(depthModelBitmapGrey);
      saveFloatArrayToBinary(greyBitmapFloatArray, String.format("batch_%d_depth_map_grey_%d.bin", batchNumber, frameNumber));
      Log.i("CollectFrames", String.format("Saved depth map grey for batch %d, frame %d", batchNumber, frameNumber));

      // CONFIDENCE POINTS
      if (transformedConfidencePoints4d != null) {
        saveFloatArrayToBinary(transformedConfidencePoints4d, String.format("batch_%d_confidence_points_%d.bin", batchNumber, frameNumber));
        Log.i("CollectFrames", String.format("Saved confidence points for batch %d, frame %d", batchNumber, frameNumber));
      }



//    }


//      waitForUser();


  }
  catch (Exception e) {
    Log.e("MyDepth", "Depth estimation failed: " + e.getMessage(), e);
  }

}

  private void runDepthEstimation(Image cameraImage, Image depthImage, Image confidenceImage) {

    try  {

            // get the image dimensions and rotation angle
      int imageWidth = cameraImage.getWidth();
      int imageHeight = cameraImage.getHeight();

      int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
      int rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);

      // Get bitmap from Camera image and then close cameraImage
      Bitmap cameraBitmap = imageToBitmap(cameraImage);
      cameraImage.close();  // as soon as we're done with it
      cameraBitmap = rotateBitmap(cameraBitmap, rotationDegrees);
      Bitmap depthModelInputBitmap = Bitmap.createScaledBitmap(cameraBitmap, 256, 256, true);

      Log.i("CollectFrames", "Camera done ...");

      // get the raw depth image, extract the depth points, and transform it to camera image space
      // and then close the images
      int depthWidth = depthImage.getWidth();
      int depthHeight = depthImage.getHeight();
      float[] points4d = getDepthPoints(depthImage, confidenceImage, 0.7f);

      Log.i("CollectFrames", "points4d done ...");

      float[] transformedPoints4d = null;
      if (points4d != null) transformedPoints4d = mapDepthPointsToCameraImage(points4d, imageWidth, imageHeight, depthWidth, depthHeight);

      Log.i("CollectFrames", "transformedPoints4d done ...");

      depthImage.close();
      confidenceImage.close();

        ByteBuffer inputBuffer = bitmapToByteBuffer(depthModelInputBitmap);           // convert bitmap to input buffer and create empty output buffer to write to
        float[][][][] outputBuffer = new float[1][256][256][1];
        tflite_interpreter.run(inputBuffer, outputBuffer);  // run MIDAS

Log.i("CollectFrames", "tflite done ...");


      ColourMapResult colourMapResult = createColorMappedBitmap(outputBuffer, 256, 256); // create coloured bitmap from Midas output buffer

        Bitmap depthModelBitmapColour = Bitmap.createScaledBitmap(colourMapResult.colourBitmap, 480, 640, true);
        Bitmap depthModelBitmapGrey = Bitmap.createScaledBitmap(colourMapResult.greyscaleBitmap, 480, 640, true);

Log.i("CollectFrames", "depthMaps done ...");

      // display the depth map as a transparent overlay over the camera image
      Bitmap combinedBitmap = cameraBitmap.copy(Bitmap.Config.ARGB_8888, true);
      Canvas canvas = new Canvas(combinedBitmap);
        Paint transparentPaint = new Paint();
        transparentPaint.setAlpha(128); // or tune as needed (0=transparent, 255=opaque)
        canvas.drawBitmap(depthModelBitmapColour, 0, 0, transparentPaint);

Log.i("CollectFrames", "draw depthMaps done ...");

if (transformedPoints4d == null) Log.i("CollectFrames", String.format("transformedPoints4d is null"));

      // draw the depth points over the camera and depth map images
      if (transformedPoints4d != null) {
        Bitmap depthPointsImage = drawPointsOverlay(transformedPoints4d, imageWidth, imageHeight);
        depthPointsImage = rotateBitmap(depthPointsImage, rotationDegrees);
        canvas.drawBitmap(depthPointsImage, 0, 0, null);
      }

Log.i("CollectFrames", "draw points done ...");

      displayBitmapWithAspectRatio(combinedBitmap);

Log.i("CollectFrames", "displayBitmapWithAspectRatio points done ...");

//      numFramesTaken++; // update it right at the end so we don't count excpetion cases
      Log.i("CollectFrames", String.format("Num Frames Taken %d", numFramesCollected -1));

      if (numFramesCollected <= 10) {

        saveFloatArrayToBinary(transformedPoints4d, String.format("depth_points_%d.bin", numFramesCollected - 1));
        Log.i("CollectFrames", String.format("Saved depth points for frame %d", numFramesCollected - 1));

        float[] cameraBitmapFloatArray = convertBitmapToFloatPointsArray(cameraBitmap);
        saveFloatArrayToBinary(cameraBitmapFloatArray, String.format("depth_map_camera_%d.bin", numFramesCollected -1));
        Log.i("CollectFrames", String.format("Saved depth map camera for frame %d", numFramesCollected -1));

        float[] colourBitmapFloatArray = convertBitmapToFloatPointsArray(depthModelBitmapColour);
        saveFloatArrayToBinary(colourBitmapFloatArray, String.format("depth_map_colour_%d.bin", numFramesCollected -1));
        Log.i("CollectFrames", String.format("Saved depth map colour for frame %d", numFramesCollected -1));

        float[] greyBitmapFloatArray = convertBitmapToFloatPointsArray(depthModelBitmapGrey);
        saveFloatArrayToBinary(greyBitmapFloatArray, String.format("depth_map_grey_%d.bin", numFramesCollected -1));
        Log.i("CollectFrames", String.format("Saved depth map grey for frame %d", numFramesCollected -1));

      }


//      waitForUser();


    }
    catch (Exception e) {
      Log.e("MyDepth", "Depth estimation failed: " + e.getMessage(), e);
    }

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

  private static float clamp(float val) {
    return Math.max(0f, Math.min(1f, val));
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

  private float[] getDepthPoints(Image depthImage, Image confidenceImage, final float confidenceLimit) {

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

  private static ShortBuffer convertDepthImageToShortBuffer(Image depth)    {

    final Image.Plane depthImagePlane = depth.getPlanes()[0];

    ByteBuffer depthByteBufferOriginal = depthImagePlane.getBuffer();
    ByteBuffer depthByteBuffer = ByteBuffer.allocate(depthByteBufferOriginal.capacity());

    depthByteBuffer.order(ByteOrder.LITTLE_ENDIAN);

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

// </editor-fold>

// <editor-fold desc="Helper Methods - images">


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

  private static ColourMapResult createColorMappedBitmap(float[][][][] depth, int width, int height) {
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
    Bitmap greyscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    // Second pass: map each value to a color
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float normalized = (depth[0][y][x][0] - min) / (max - min);  // ∈ [0, 1]
        int color = infernoColormap(normalized);  // Choose colormap here
        bitmap.setPixel(x, y, color);

        int grey = Math.round(normalized * 255.0f);
        grey = Math.max(0, Math.min(255, grey)); // Clamp
        int argb = Color.argb(255, grey, grey, grey);  // Full alpha, grayscale RGB
        greyscaleBitmap.setPixel(x, y, argb);
      }
    }

    return new ColourMapResult(bitmap, greyscaleBitmap);
  }

  public static class ColourMapResult {
    public final Bitmap colourBitmap;
    public final Bitmap greyscaleBitmap;

    public ColourMapResult(Bitmap colorBitmap, Bitmap greyscaleBitmap) {
      this.colourBitmap = colorBitmap;
      this.greyscaleBitmap = greyscaleBitmap;
    }
  }

// </editor-fold>

// <editor-fold desc="Helper methods - display bitmaps">
  public void displayBitmapWithAspectRatio(Bitmap bitmap) {
    if (bitmap == null || bitmapView == null) return;

//    clearBitmapView();
    bitmapView.setAdjustViewBounds(true);
    bitmapView.setScaleType(ImageView.ScaleType.FIT_CENTER);
    bitmapView.setAlpha(1.0f);
    bitmapView.setVisibility(View.VISIBLE);
    bitmapView.setImageBitmap(bitmap);
//    bitmapView.bringToFront(); // ensure it's above GLSurfaceView
  }

  public void clearBitmapView() {
    if (bitmapView == null) return;

    runOnUiThread(() -> {
      bitmapView.setImageBitmap(null);
      bitmapView.setVisibility(View.GONE);
    });
  }
// </editor-fold>

// <editor-fold desc="Helper methods - bitmap transformations">
  private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
    Matrix matrix = new Matrix();
    matrix.postRotate(rotationDegrees);
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
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

// <editor-fold desc="Helper methods - for Midas">
  private MappedByteBuffer loadModelFile(AppCompatActivity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
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

// </editor-fold>

// <editor-fold desc="Helper methods - output">
  private void saveFloatArrayToBinary(float[] array, String filename) {

    if (array == null) return;

    try {
      File path = new File(getExternalFilesDir(null), "exported");
      if (!path.exists()) path.mkdirs();

      File file = new File(path, filename);
      FileOutputStream fos = new FileOutputStream(file);
      ByteBuffer buffer = ByteBuffer.allocate(array.length * 4);
      buffer.order(ByteOrder.LITTLE_ENDIAN); // Match expected order on PC
      for (float f : array) buffer.putFloat(f);
      fos.write(buffer.array());
      fos.close();

      Log.i("ExportFloat", "Saved to: " + file.getAbsolutePath());
    } catch (IOException e) {
      Log.e("ExportFloat", "Failed to write binary float array", e);
    }
  }
  public static ByteBuffer bitmapToByteBufferForOutput(Bitmap bitmap) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    Bitmap.Config config = bitmap.getConfig();

    ByteBuffer buffer;

    if (config == Bitmap.Config.ARGB_8888) {
      // 4 bytes per pixel
      buffer = ByteBuffer.allocate(width * height * 4);
    } else if (config == Bitmap.Config.ALPHA_8) {
      // 1 byte per pixel
      buffer = ByteBuffer.allocate(width * height);
    } else {
      throw new IllegalArgumentException("Unsupported bitmap config: " + config);
    }

    buffer.order(ByteOrder.nativeOrder());
    bitmap.copyPixelsToBuffer(buffer);
    buffer.rewind();  // Reset position to 0 so it's ready for reading
    return buffer;
  }
  private void saveByteBufferToBinary(ByteBuffer buffer, String filename) {
    try {
      File path = new File(getExternalFilesDir(null), "exported");
      if (!path.exists()) path.mkdirs();

      File file = new File(path, filename);
      FileOutputStream fos = new FileOutputStream(file);

      fos.write(buffer.array());
      fos.close();

      Log.i("ExportByteBuffer", "Saved to: " + file.getAbsolutePath());
    } catch (IOException e) {
      Log.e("ExportByteBuffer", "Failed to write binary ByteBuffer", e);
    }
  }
  private static float[] convertBitmapToFloatPointsArray(Bitmap bitmap) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    float[] result = new float[width * height * 6];

    int index = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = bitmap.getPixel(x, y);

        int a = (pixel >> 24) & 0xff;
        int r = (pixel >> 16) & 0xff;
        int g = (pixel >> 8) & 0xff;
        int b = pixel & 0xff;

        result[index++] = (float) x;
        result[index++] = (float) y;
        result[index++] = a / 255.0f;
        result[index++] = r / 255.0f;
        result[index++] = g / 255.0f;
        result[index++] = b / 255.0f;
      }
    }

    return result;
  }
// </editor-fold>

// <editor-fold desc="Helper methods - Colour maps">
  private static int rainbowColormap(float x) {
    x = Math.max(0f, Math.min(1f, x));  // Clamp to [0, 1]

    // Simple piecewise triangular functions to simulate rainbow
    float r = clamp(1.5f - Math.abs(4f * x - 1f));
    float g = clamp(1.5f - Math.abs(4f * x - 2f));
    float b = clamp(1.5f - Math.abs(4f * x - 3f));

    return Color.argb(255, (int)(r * 255), (int)(g * 255), (int)(b * 255));
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
// </editor-fold>

  private void waitForUser() {
    // 1. Show the button and set up the click listener
    runOnUiThread(() -> {
      continueButton.setVisibility(View.VISIBLE);
      continueButton.setEnabled(true);
      continueButton.setOnClickListener(v -> {
        synchronized (continueLock) {
          continuePressed = true;
          continueLock.notify();
        }
      });
    });

    // 2. Block the GL thread until the button is pressed
    synchronized (continueLock) {
      while (!continuePressed) {
        try {
          continueLock.wait();
        } catch (InterruptedException e) {
          Log.e("DepthPause", "Interrupted while waiting for user", e);
        }
      }
      continuePressed = false;  // reset for next time
    }

    // 3. Hide the button again
    runOnUiThread(() -> continueButton.setVisibility(View.GONE));
  }

  public static class FrameData {

    //<editor-fold desc="FrameData member variables">

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

    private float[] getConfidencePoints() {

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
        
    private float[] getDepthPoints(final float confidenceLimit) {

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

}





