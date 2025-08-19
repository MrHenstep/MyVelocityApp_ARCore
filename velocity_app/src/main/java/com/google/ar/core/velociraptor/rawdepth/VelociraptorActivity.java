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

package com.google.ar.core.velociraptor.rawdepth;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;

import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.graphics.PointF;

import android.widget.ImageView;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import android.view.MotionEvent;
import android.view.View;

import android.widget.Button;
import android.widget.Toast;


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
import com.google.ar.core.velociraptor.common.helpers.CameraPermissionHelper;
import com.google.ar.core.velociraptor.common.helpers.DisplayRotationHelper;
import com.google.ar.core.velociraptor.common.helpers.FullScreenHelper;
import com.google.ar.core.velociraptor.common.helpers.SnackbarHelper;

import com.google.ar.core.velociraptor.common.rendering.TextureRenderer;


import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

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
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.video.Video;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;


public class VelociraptorActivity extends AppCompatActivity implements GLSurfaceView.Renderer {

// <editor-fold desc="Member Variables">
  private static final String TAG = VelociraptorActivity.class.getSimpleName();

  // Views
  private GLSurfaceView glSurfaceView;  // needs to be resumed/paused etc. because it live on GL rendering thread
  private ImageView previewView; // doesn't need resumed/paused etc. because it is passive

  // Renderers
  // *** The backgroundRenderer neatly binds the camera feed to a texture etc.
  // *** Probably we could extract this but as it works, we just turn off the drawing for now
  private final TextureRenderer backgroundRenderer = new TextureRenderer();


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
  private static final int FRAME_THROTTLE_INTERVAL = 6;
  private static final int MAX_COLLECTED_FRAMES = 10;


  private Button continueButton;
//  private final Object continueLock = new Object();
//  private boolean continuePressed = false;

  private long baseTimestamp;

  private volatile Bitmap lastPreviewBitmap = null;

  private Mat prevGrey = null;

  // For point-tracking
//  private org.opencv.core.Mat previousFrameGreyScale = null;
  private final AtomicReference<PointF> trackingPointRef =
        new AtomicReference<>(new PointF(0.0f, 0.0f));



//  private boolean showLiveDepthPoints = true; // draw depth points even when not collecting

  private final AtomicBoolean previewBusy = new AtomicBoolean(false);


  // Cached collection of frames
  private final List<CollectedSample> collectedFramesContainer = Collections.synchronizedList(new ArrayList<>());

  private volatile boolean collectingFrames = false;

  private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
  private final ExecutorService previewExecutor = Executors.newSingleThreadExecutor();

  private final AtomicBoolean processingFrames = new AtomicBoolean(false);

  private int batchCounter = 0;  // starts at 0

  // Member variable to hold last tapped view coords
//  private PointF tappedPointViewCoords = null;


  // Member variables
  private volatile int latestBitmapWidth = 0;
  private volatile int latestBitmapHeight = 0;


// </editor-fold>

// <editor-fold desc="Activity Lifecycle">

  @SuppressLint("ClickableViewAccessibility")
  @Override
  protected void onCreate(Bundle savedInstanceState) {

    // housekeeping
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    initialiseViews();

    // Assume we don't need to install
    installRequested = false;

    // make sure the bitmap view is on the top
    previewView.bringToFront();

    initialiseContinueButton();

    initialiseTouchListener();

    // (try to) load the model
    try {
      tflite_interpreter = new Interpreter(loadModelFile(this));
    } catch (IOException e) {
      Log.e("TFLite", "Error loading TFLite model: ", e);
    }

    System.loadLibrary("opencv_java4");

//    prevGrey = new Mat();
  }

  @SuppressLint("ClickableViewAccessibility")
  private void initialiseTouchListener() {

    // listen for a screen-tap to set the point to be tracked

    previewView.setOnTouchListener((view, event) -> {
      if (event.getAction() == MotionEvent.ACTION_DOWN) {
        if (latestBitmapWidth == 0 || latestBitmapHeight == 0) {
          Log.w("Touch", "Bitmap size unknown at tap time, coordinate mapping might be off.");
          return false;
        }

        // Map to bitmap coordinates
        PointF mapped = mapTouchEventToBitmapPoint(
                event,
                (ImageView) view,
                latestBitmapWidth,
                latestBitmapHeight
        );

        float bitmapX = mapped.x;
        float bitmapY = mapped.y;

        trackingPointRef.set(new PointF(bitmapX, bitmapY));
        view.performClick();
        return true;
      } else {
        return false;
      }
    });

  }
  @SuppressLint("SetTextI18n")
  private void initialiseContinueButton() {

    continueButton = findViewById(R.id.continue_button);
    continueButton.setVisibility(View.VISIBLE);  // Show the button on app start

    // listen for the "CONTINUE" button to be pressed.
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
  }
  private void initialiseViews() {

    // set up the member variable view IDs
    glSurfaceView = findViewById(R.id.surfaceview);
    previewView = findViewById(R.id.preview_view);

    // instantiate the member display rotation helper
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    // Set up the glSurfaceView, which will be used for the rendering
    glSurfaceView.setPreserveEGLContextOnPause(true);
    glSurfaceView.setEGLContextClientVersion(2);
    glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
    glSurfaceView.setRenderer(this);
    glSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY); // tells GL thread to call this once per frame
    glSurfaceView.setWillNotDraw(false);
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

        // check and if necessary request camera permission
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
    previewExecutor.shutdown();
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

    // We need a renderer to render camera image to a texture for ARCore to use
    try {
      backgroundRenderer.createOnGlThread(/*context=*/ this);
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
  @SuppressLint("SetTextI18n")
  @Override
  public void onDrawFrame(GL10 gl) {

    if (session == null) return;

    displayRotationHelper.updateSessionIfNeeded(session);
    session.setCameraTextureName(backgroundRenderer.getTextureId());
    int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
    int rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);

    // ********************************************************************************
    // *** Get the Frame, and build the FrameData                                   ***
    // ********************************************************************************

    Frame frame;
    FrameData frameData = null;

    try {
      frame = session.update();
      frameData = new FrameData(frame);
    }
    catch (Throwable t) {
      Log.e("onDraw", "FrameData build failed", t);
    }

    if (trackingPointRef.get() != null && trackingPointRef.get().x < 1f && trackingPointRef.get().y < 1f) {
      trackingPointRef.set(new PointF(frameData.cameraHeight * 0.5f, frameData.cameraWidth * 0.5f));
    }

    displayPreview(frameData, rotationDegrees);

    // ********************************************************************************
    // *** Check for break conditions, e.g. if we're not collecting, or are done    ***
    // ********************************************************************************

    // if we're not collecting frames, close the images safely and return
    if (!collectingFrames) {return; }

    // if we're throttling and rejecting this frame, close the images safely and return.
    numFramesTaken++;
    if (numFramesTaken % FRAME_THROTTLE_INTERVAL != 0) {return; }


    // *************************************************************************************
    // *** Process the frame - store it, and if we're done, set of the processing thread ***
    // *************************************************************************************

    try {

      if (frameData != null && frameData.isValid) {


        runOnUiThread(() -> {
          continueButton.setText("Collecting: Batch " + (batchCounter-1) + " Frame " + (numFramesCollected-1) + "/" + (MAX_COLLECTED_FRAMES-1));
        });
        numFramesCollected++;

        if (numFramesCollected == 1) baseTimestamp = frameData.frameTimestamp;
        frameData.baseTimestamp = baseTimestamp;

        // synchronise this section
        synchronized (collectedFramesContainer) {

          // store the new frameData
          PointF tapSnapshot = null;
          PointF tp = trackingPointRef.get();   // volatile read
          if (tp != null) tapSnapshot = new PointF(tp.x, tp.y);

          collectedFramesContainer.add(new CollectedSample(frameData, tapSnapshot));

          // if we've collected the whole batch, set off the processing thread
          if (collectedFramesContainer.size() >= MAX_COLLECTED_FRAMES) {

            Log.i("CollectFrames", collectedFramesContainer.size() + " Frames Collected");

            final int thisBatch = batchCounter - 1;

            // notify the user that we're processing
//            runOnUiThread(() -> {
//              continueButton.setText("Processing: Batch " + thisBatch);
//              continueButton.setEnabled(false);
//              continueButton.setVisibility(View.VISIBLE);
//              continueButton.bringToFront();
//              continueButton.setElevation(getResources().getDisplayMetrics().density * 16f); // ~16dp
//            });

            collectingFrames = false;
            processingFrames.set(true);

            // copy the collected frames into a list and clear the list
            // this is so that the processing, which is about to execute on a different thread,
            // can work on it while the GL thread carries on displaying previews
            // (just in case onDrawFrame is ever changed to collect while processing)
//            List<FrameData> snapshot = new ArrayList<>(collectedFramesContainer);
            List<CollectedSample> snapshot = new ArrayList<>(collectedFramesContainer);
            collectedFramesContainer.clear();
            numFramesCollected = 0;


            // run the number-crunching on an executor so we don't block either the GL or the UI thread
            analysisExecutor.execute(() -> {

              try {
                PointTrackingState prevPointTrackingState = null;
                for (int iFrame = 0; iFrame < snapshot.size(); ++iFrame) {

                  final int frameNumber = iFrame;
                  runOnUiThread(() -> {
                    continueButton.setText("Processing: Batch " + thisBatch + " Frame " + frameNumber + "/" + (MAX_COLLECTED_FRAMES-1));
                  });

                  CollectedSample sample = snapshot.get(iFrame);
                  if (iFrame == 0) prevPointTrackingState = new PointTrackingState(sample.tap, null);

                  prevPointTrackingState = processCollectedFrameData(sample.frame, iFrame, thisBatch, prevPointTrackingState, sample.tap);


                }
              } finally {
                processingFrames.set(false);
                runOnUiThread(() -> {
                  // Ready for another round: make the button usable again.
                  continueButton.setText("Collect Frames");
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

// <editor-fold desc="display previews etc."

  private void displayPreview(FrameData frameData, int rotationDegrees) {
    if (frameData != null && frameData.isValid) {
      // Only queue one preview job at a time; drop stale frames.
      if (previewBusy.compareAndSet(false, true)) {
        previewExecutor.execute(() -> {
          try {
            renderCameraAndPointsFromFrameData(frameData, rotationDegrees);
          } finally {
            previewBusy.set(false);
          }
        });
      }
    }
  }

  private void renderCameraAndPointsFromFrameData(FrameData frameData, int rotationDegrees) {

    // get the camera bitmap for the preview
    Bitmap preview = frameData.getCameraBitmap();


    // if not available, use the last stored one, and return
    if (!frameData.isValid || preview == null) {
      if (lastPreviewBitmap != null) {
        Bitmap finalBmp = lastPreviewBitmap;
        runOnUiThread(() -> displayBitmapWithAspectRatio(finalBmp));
      }
      return;
    }

    // if we have a bitmap, compose preview with depth points on top
    preview = rotateBitmap(preview, rotationDegrees);
    lastPreviewBitmap = preview;  // update the stored version
    Bitmap composed = preview;
    composed = drawDepthPoints(composed, frameData, rotationDegrees, frameData.cameraWidth, frameData.cameraHeight, 0.7f);


    // now run the point tracking
    Mat grey = toGreyScaleMat(preview);

    PointF prevPoint = trackingPointRef.get();
    PointF newPoint = trackPoint(prevGrey, grey, prevPoint);
    if (newPoint != null) trackingPointRef.set(newPoint);
    prevGrey = grey.clone();

    grey.release();

    // and draw the new point on the composed bitmap
    composed = drawTrackedPoint(composed, trackingPointRef.get());

    // finally, push it all to the UI thread for display
    Bitmap finalBmp = composed;
    runOnUiThread(() -> displayBitmapWithAspectRatio(finalBmp));

    latestBitmapWidth = finalBmp.getWidth();
    latestBitmapHeight = finalBmp.getHeight();


  }

  private PointF trackPoint(Mat prevGrey, Mat grey, PointF prevPoint) {

    PointF newPoint = null;

    if (prevGrey == null) return null;

    MatOfPoint2f prevPt = new MatOfPoint2f(new org.opencv.core.Point(prevPoint.x, prevPoint.y));
    MatOfPoint2f nextPt = new MatOfPoint2f();
    MatOfByte status = new MatOfByte();
    MatOfFloat err = new MatOfFloat();

    Video.calcOpticalFlowPyrLK(
            prevGrey, grey, prevPt, nextPt, status, err,
            new org.opencv.core.Size(21,21), 3);

    byte[] st = status.toArray();
    if (st.length > 0 && (st[0] & 0xFF) == 1) {
      org.opencv.core.Point p = nextPt.toArray()[0];
      newPoint = new PointF((float)p.x, (float)p.y);
    }

    prevPt.release(); nextPt.release(); status.release(); err.release();
    prevGrey.release();

    return newPoint;
  }

  private Mat toGreyScaleMat(Bitmap bitmap) {
    Mat frame = new Mat();
    Utils.bitmapToMat(bitmap, frame);

    Mat grey = new Mat();
    Imgproc.cvtColor(frame, grey, Imgproc.COLOR_RGBA2GRAY);

    return grey;
  }

  private Bitmap drawDepthPoints(Bitmap baseRotated, FrameData frameData, int rotationDegrees, int camW, int camH, float confidenceLimit) {

    if (baseRotated == null || frameData == null || !frameData.isValid) return baseRotated;

    try {
      float[] pointsToMap = frameData.getDepthPoints(confidenceLimit);
      if (pointsToMap == null || pointsToMap.length < 4) return baseRotated;

      float[] mappedPoints = frameData.mapDepthPointsToCameraImage(pointsToMap);

      // Create overlay in pre-rotation space, then rotate to display orientation
      Bitmap overlay = drawPointsOverlay(mappedPoints, camW, camH);
      overlay = rotateBitmap(overlay, rotationDegrees);

      // Draw onto a mutable copy
      Bitmap composed = ensureMutable(baseRotated);
      Canvas c = new Canvas(composed);
      c.drawBitmap(overlay, 0, 0, null);
      return composed;

    } catch (Throwable t) {
      Log.w(TAG, "Live depth overlay failed; showing preview only.", t);
      return baseRotated;
    }
  }

  private Bitmap drawTrackedPoint(Bitmap source, PointF p) {
    if (source == null || p == null) return source;

    int w = source.getWidth();
    int h = source.getHeight();
    float x = p.x;
    float y = p.y;

    // Bounds check
    if (x < 0 || x >= w || y < 0 || y >= h) return source;

    // Ensure we draw on a mutable copy
    Bitmap composed = ensureMutable(source);

    Canvas canvas = new Canvas(composed);
    Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setAntiAlias(true);

    // Cross size relative to image
    float s = Math.min(w, h);
    float halfLen = Math.max(6f, 0.02f * s);
    float stroke = Math.max(2f, 0.003f * s);
    paint.setStrokeWidth(stroke);

    // Draw cross
    canvas.drawLine(x - halfLen, y, x + halfLen, y, paint);
    canvas.drawLine(x, y - halfLen, x, y + halfLen, paint);

    return composed;
  }

  private static Bitmap ensureMutable(Bitmap src) {
    if (src == null) return null;
    return src.isMutable() ? src : src.copy(Bitmap.Config.ARGB_8888, true);
  }

  public void displayBitmapWithAspectRatio(Bitmap bitmap) {

    if (bitmap == null || previewView == null) return;

    previewView.setAdjustViewBounds(true);
    previewView.setScaleType(ImageView.ScaleType.FIT_CENTER);
    previewView.setAlpha(1.0f);
    previewView.setVisibility(View.VISIBLE);
    previewView.setImageBitmap(bitmap);
  }

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
        int argb = Color.argb(255, grey, grey, grey);  // Full alpha, greyscale RGB
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

  private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
    Matrix matrix = new Matrix();
    matrix.postRotate(rotationDegrees);
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
  }

// </editor-fold>

// <editor-fold desc="Main processCollectedFrameData method">
  @SuppressLint("DefaultLocale")
  private PointTrackingState processCollectedFrameData(FrameData frameData, int frameNumber, int batchNumber, PointTrackingState prevState, PointF estimatedPoint) {

  PointTrackingState newState = null;

  Log.i("processCollectedFrameData", String.format("tracked for batch %d, frame %d", batchNumber, frameNumber));

  try  {

    // -------------------------------------------------------------------------------------------------------------------
    // get the depth and confidence points

    float[] points4d = frameData.getDepthPoints(0.5f);
    float[] transformedPoints4d = null;
    if (points4d != null) transformedPoints4d = frameData.mapDepthPointsToCameraImage(points4d);

    float[] confidencePoints4d = frameData.getConfidencePoints();
    float[] transformedConfidencePoints4d = null;
    if (confidencePoints4d != null) transformedConfidencePoints4d = frameData.mapDepthPointsToCameraImage(confidencePoints4d);

    // -------------------------------------------------------------------------------------------------------------------
    // get camera image and rotate

    int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
    int rotationDegrees = getCameraImageRotationDegrees(this, displayRotation);

    Log.i("ROTATION", String.format("Display: %d, Rotation: %d", displayRotation, rotationDegrees));

    Bitmap cameraBitmap = frameData.getCameraBitmap();

    // -------------------------------------------------------------------------------------------------------------------
    // run depth estimation on the camera image

    Bitmap rotatedCameraBitmap = rotateBitmap(cameraBitmap, rotationDegrees);;

    final int midasWidth = 256;
    final int midasHeight = 256;
    Bitmap depthModelInputBitmap = Bitmap.createScaledBitmap(rotatedCameraBitmap, midasWidth, midasHeight, true);

    ByteBuffer inputBuffer = bitmapToByteBuffer(depthModelInputBitmap);           // convert bitmap to input buffer and create empty output buffer to write to
    float[][][][] outputBuffer = new float[1][256][256][1];
    tflite_interpreter.run(inputBuffer, outputBuffer);  // run MIDAS

    ColourMapResult rotatedColourMapResult = createColorMappedBitmap(outputBuffer, midasWidth, midasHeight); // create coloured bitmap from Midas output buffer

    final int imageWidth = rotatedCameraBitmap.getWidth();
    final int imageHeight = rotatedCameraBitmap.getHeight();
    Bitmap rotatedDepthModelBitmapColour = Bitmap.createScaledBitmap(rotatedColourMapResult.colourBitmap, imageWidth, imageHeight, true);
    Bitmap rotatedDepthModelBitmapGrey = Bitmap.createScaledBitmap(rotatedColourMapResult.greyscaleBitmap, imageWidth, imageHeight, true);

    Bitmap depthModelBitmapColour = rotateBitmap(rotatedDepthModelBitmapColour, -rotationDegrees);
    Bitmap depthModelBitmapGrey = rotateBitmap(rotatedDepthModelBitmapGrey, -rotationDegrees);



    // -----------------------------------------------------------------------------------------------------
    // run point tracker
    Mat newGrey = toGreyScaleMat(cameraBitmap);
    PointF newPoint = null;
    if (prevState != null && prevState.prevGrey != null) {
      // normal operation - use the previous grey with the new one, and update the previous point
      newPoint = trackPoint(prevState.prevGrey, newGrey, prevState.prevPoint);
      newState = new PointTrackingState(newPoint, newGrey.clone());
    }
    else {
      // first call - no previous grey, so can't track, just update the prevState with the new one
      PointF prevPointInCameraFrame = rotatePointF(prevState.prevPoint, -rotationDegrees, latestBitmapWidth, latestBitmapHeight);
      newState = new PointTrackingState(prevPointInCameraFrame, newGrey.clone());
    }
    newGrey.release();

    // -----------------------------------------------------------------------------------------------------
    // now save all to file

    // TIMESTAMPS
    float[] timeStamps = new float[] {
            frameData.frameTimestamp - frameData.baseTimestamp,
            frameData.cameraTimestamp - frameData.baseTimestamp,
            frameData.depthTimestamp - frameData.baseTimestamp,
            frameData.confidenceTimestamp - frameData.baseTimestamp
    };
    saveFloatArrayToBinary(timeStamps, String.format("batch_%d_timestamps_%d.bin", batchNumber, frameNumber));
    Log.i("CollectFrames", String.format("Saved timestamps for batch %d, frame %d", batchNumber, frameNumber));

    //     DEPTH POINTS
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

    // TEXTURE INTRINSICS
    float[] textureIntrinsics = frameData.textureIntrinsicsToFloatArray();
    saveFloatArrayToBinary(textureIntrinsics, String.format("batch_%d_texture_intrinsics_%d.bin", batchNumber, frameNumber));

    // CAMERA INTRINSICS
    float[] cameraIntrinsics = frameData.cameraIntrinsicsToFloatArray();
    saveFloatArrayToBinary(cameraIntrinsics, String.format("batch_%d_camera_intrinsics_%d.bin", batchNumber, frameNumber));

    // EXTRINSIC MATRIX
    float[] extrinsicMatrix = frameData.extrinsicMatrixHomToFloatArray();
    saveFloatArrayToBinary(extrinsicMatrix, String.format("batch_%d_extrinsic_matrix_%d.bin", batchNumber, frameNumber));

    // TRACKED POINTS
    float newX = (newPoint != null ? newPoint.x : newState.prevPoint.x);
    float newY = (newPoint != null ? newPoint.y : newState.prevPoint.y);
    float[] pointCoordinates = new float[] {newX, newY, estimatedPoint.x, estimatedPoint.y};
    saveFloatArrayToBinary(pointCoordinates, String.format("batch_%d_tracked_point_%d.bin", batchNumber, frameNumber));

    Log.i("processCollectedFrameData", String.format("tracked for batch %d, frame %d: %f, %f, %f, %f", batchNumber, frameNumber, newX, newY, estimatedPoint.x, estimatedPoint.y));

  }
  catch (Exception e) {
    Log.e("MyDepth", "Depth estimation failed: " + e.getMessage(), e);
  }

  return newState;
}

// </editor-fold>

// <editor-fold desc="Helper Methods & Classes">

  private static float clamp(float val) {
    return Math.max(0f, Math.min(1f, val));
  }

  private PointF mapTouchEventToBitmapPoint(MotionEvent event, ImageView imageView,
                                            int bitmapWidth, int bitmapHeight) {

    Drawable drawable = imageView.getDrawable();
    if (drawable == null) {
      // No drawable, return touch coords in view space
      return new PointF(event.getX(), event.getY());
    }

    Matrix imageMatrix = imageView.getImageMatrix();

    // Get intrinsic dimensions of the drawable (i.e., bitmap)
    int drawableWidth = drawable.getIntrinsicWidth();
    int drawableHeight = drawable.getIntrinsicHeight();

    if (drawableWidth <= 0 || drawableHeight <= 0) {
      return new PointF(event.getX(), event.getY());
    }

    // Invert the matrix
    Matrix inverseMatrix = new Matrix();
    imageMatrix.invert(inverseMatrix);

    // Map touch point from view coords → bitmap coords
    float[] touchPoint = new float[] { event.getX(), event.getY() };
    inverseMatrix.mapPoints(touchPoint);

    float bitmapX = touchPoint[0];
    float bitmapY = touchPoint[1];

    // Clamp to bitmap bounds
    bitmapX = Math.max(0f, Math.min(bitmapX, (float) bitmapWidth));
    bitmapY = Math.max(0f, Math.min(bitmapY, (float) bitmapHeight));

    return new PointF(bitmapX, bitmapY);
  }

  // Put near other inner classes / members
  private static final class CollectedSample {
    final FrameData frame;
    final PointF tap; // in rotated-bitmap pixel coords (same space as your preview)

    CollectedSample(FrameData frame, PointF tap) {
      this.frame = frame;
      this.tap = (tap == null) ? null : new PointF(tap.x, tap.y); // defensive copy
    }
  }

  private static final class PointTrackingState {
    final PointF prevPoint;
    final Mat prevGrey;

    PointTrackingState(PointF point, Mat grey) {
      this.prevPoint = point;
      this.prevGrey = grey;
    }
  }
//
// </editor-fold>

// <editor-fold desc="Helper Methods - orientation">

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
          return (sensorOrientation - deviceRotationDegrees + 360) % 360;
        }
      }
    } catch (CameraAccessException e) {
      Log.e("CameraRotation", "Failed to access camera characteristics", e);
    }

    return 0;
  }


  public static PointF rotatePointF(PointF p, int angleClockwise, int originalWidth, int originalHeight) {
    int a = ((angleClockwise % 360) + 360) % 360;
    switch (a) {
      case 0:
        return new PointF(p.x, p.y);
      case 90:  // (x,y) -> (h-1 - y, x)
        return new PointF(originalHeight - 1 - p.y, p.x);
      case 180: // (x,y) -> (w-1 - x, h-1 - y)
        return new PointF(originalWidth - 1 - p.x, originalHeight - 1 - p.y);
      case 270: // (x,y) -> (y, w-1 - x)
        return new PointF(p.y, originalWidth - 1 - p.x);
      default:
        throw new IllegalArgumentException("angleCW must be 0, 90, 180, or 270");
    }
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

}





