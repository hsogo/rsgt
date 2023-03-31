# rsgt README

rsgt is a python module for gaze recording using a single visible-light camera such as a webcam.  Please note that gaze estimation from a single visible-light camera under uncontrolled lighting conditions is a quite challenging task so we don't expect much spatial accuracy and precision.

- **rsgt.app.tracker**: An application for live gaze tracking.  It captures camera images, detects faces, and calculates gaze direction.  Because face detection and gaze calculation take tens of milliseconds, the sampling frequency would not be less than the camera fps (tested on Core i7 11700 @2.5GHz).
- **rsgt.app.offline_tracker**: This is basically the same as rsgt.tracker, but capture images from a movie file.  It can handle movie frames without dropping (will take time longer than the movie).
- **rsgt.app.camera_calibration**: This performs camera calibration.  Camera calibration is necessary to estimate face location and posture relative to the camera.
- **rsgt.app.offline_calibration**: This calculates mapping from gaze vector to screen location from a movie file.  It is intended to use with rsgt.offline_tracker.

## Setup

On the first run, rsgt creates a directory named ".rsgt" in the home directory and creates three configuration files (camera_param.cfg, face_model.cfg, and rsgt.cfg) in it.  It is important to customize these files in order to make rsgt work.

### Camera calibration (camera_param.cfg)

Camera calibration must be performed once before starting measurements.  In the calibration process, you have to capture several images of a black-white chessboard with the camera you plan to use for the measurements.  A sample of the chessboard is included in this package (rsgt/resources/checker8x5.pdf).  If you use this chessboard for camera calibration, print it to 100% size and stick it on a flat and stiff board.
To start camera calibration, execute following command.

```
python -m rsgt.app.camera_calibration
```

Next, add images of the chessboard from *Add images from...* on the menu bar.  There are three ways to add images.

- Camera: You can specify Camera ID (0, 1, 2,... assigned by the OpenCV library), image resolution, and start live preview.  Then, place the chessboard in front of the camera and *Add image* button.  Note that you must add several images with different chessboard poses.
- Movie file: If you already have a movie file that captures various poses of the chessboard, this method would be suitable.  You can open the video file and select the frames you want to use for calibration with a slider UI.
- Image files: If you have a set of image files of the chessboard in advance, you can add them by this method.

After adding images, you can run calibration from *Calibration* on the menu bar.  If you use the sample chessboard (rsgt/resources/checker8x5.pdf) to get images, you can simply select *Run* to start calibration.  Otherwise, you have to select *Set chessboard pattern* and input the size of your chessboard before starting calibration.

When the calibration is finished, a dialog appears to show the result.  Here is an example of the result.

```
RMS
1.1746014908307345

K
[[1.29268643e+03 0.00000000e+00 9.35109014e+02]
 [0.00000000e+00 1.30554750e+03 6.31712600e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

d
[[ 0.02382548  0.29295886  0.0099289  -0.0014157  -0.82699847]]
```

Clicking *Copy to clipboard (Config format)* button on the dialog, the result is copied in rsgt configuration file format.  Here is an example.  Open camera_param.cfg in .rsgt directory and paste this.  Note that **[Screen Layout Parameters]** section is empty in the result.  If you have already set **[Screen Layout Parameters]** to camera_param.cfg, don't overwrite them.

```
[Basic Parameters]
CAMERA_ID = 
RESOLUTION_HORIZ = 1920
RESOLUTION_VERT = 1080
DOWNSCALING = 0.5

[Calibration Parameters]
CAMERA_MATRIX_R0C0=1292.6864278606479
CAMERA_MATRIX_R0C1=0.0
CAMERA_MATRIX_R0C2=935.1090135486444
CAMERA_MATRIX_R1C0=0.0
CAMERA_MATRIX_R1C1=1305.5474960404545
CAMERA_MATRIX_R1C2=631.7125997168425
CAMERA_MATRIX_R2C0=0.0
CAMERA_MATRIX_R2C1=0.0
CAMERA_MATRIX_R2C2=1.0
DIST_COEFFS_R0C0=0.023825479232907943
DIST_COEFFS_R1C0=0.29295886400368276
DIST_COEFFS_R2C0=0.009928896187259038
DIST_COEFFS_R3C0=-0.0014156958717122172
DIST_COEFFS_R4C0=-0.0014156958717122172

[Screen Layout Parameters]
WIDTH=
HORIZ_RES=
OFFSET_X=
OFFSET_Y=
OFFSET_Z=
ROT_X=
ROT_Y=
ROT_Z=
```

### Monitor Layout (camera_param.cfg)

In **[Screen Layout Parameters]** section of camera_param.cfg, you have to set the physical screen width, horizontal resolution, and camera position relative to the screen center.
rsgt package has a tiny tool to support this task.
Execute following command to use this tool.

```
python -m rsgt.tools.screen_layout
```

A full-screen dialog is presented on the screen and a black cross appears on a white background.  Measure the screen width and camera position according to the diagram on the right and enter them in the edit boxes on the left.  Clicking *OK* button below the edit boxes, input values are shown in the rsgt configuration file format.  Here is an example.

'''
[Screen Layout Parameters]
WIDTH=510
HORIZ_RES=1920
OFFSET_X=0
OFFSET_Y=212
OFFSET_Z=10
ROT_X=-10
ROT_Y=0.0
ROT_Z=0.0
'''

Copy these lines (you can use *Copy to clipboard (Config format)* button) and paste to **[Screen Layout Parameters]** session of the camera_param.cfg.


### Face Model (face_model.cfg)

face_model.cfg defines 3D location of face landmarks.  By default, average shape of adult Japanese faces.  Better measurement results can be expected if the values are adjusted according to actual shape of the participant's face.

### Others (rsgt.cfg)

Other application settings are defined in rsgt.cfg in .rsgt directory.

- Iris detector: This parameter determines which method is used to detect irises.  Currently, "ert" and "enet" are supported. "ert" is faster, while "enet" is more accurate. **tensorflow** package is required to use "enet".
- Calibrated/Calibrationless output: rsgt use several assumptions to estimate gaze direction, therefore considerable measurement error is to be expected if the assumptions are invalid.  This error can be reduced by performing calibration using eye images when looking at targets presented at known locations.  In rsgt.cfg file, you can select whether rsgt records calibrated and uncalibrated (calibrationless) values or not.  See *Calibration for gaze direction* section for detail of the calibration process.
- Datafile open mode: This parameter determines what to do if a file with the same name already exists when you try to open a data file. *"new"*, *"rename"*, and *"overwrite"* are accepted.  *"new"* means that a data file can be opened only when a file with the same name doesn't exist.


## Measurment

rsgt has two "tracker" applications for measurement.  *Live tracker* processes live stream from the camera in real-time.  *Offline tracker* reads video stream from a file.  It is highly recommended to use **PsychoPy** and **GazeParser.TrackingTools** to control the Live tracker.

### Live tracker

*Command*

```
python -m rsgt.app.tracker
```

Live tracker accepts commands via TCP/IP connection.  The commands are compatible with **TrackingTools** module of the GazeParser (GazeParser.TrackingTools).  For example, `connect()` makes TCP/IP connection between the Live tracker and your code (for stimulus presentation), and `calibrationLoop()` starts the calibration process.  `startRecording()` starts recording gaze and `getEyePosition()` gets the current gaze position from the Live tracker.  Please see GazeParser documentation for detail.

### Offline tracker

*Command*

```
python -m rsgt.app.offline_tracker
```

Offline tracker is intended to measure gaze from movie files that have been recorded in advance.  To apply calibration for gaze direction, calibration must be performed separately using **rsgt.app.offline_calibration** and save the result to a file.  *Batch mode* is supported to set all required parameters by command line options and start measurement automatically.  This mode is helpful when you want to process many movie files at once.  Try `python -m rsgt.app.offline_tracker --help` for the help of command line options.


## Calibration for gaze direction

### for Live tracker

Calibration process can be activated by `calibrationLoop()` of **GazeParser.Tracker**.  A target is presented on the screen and the participant moves his/her eyes to follow the target's motion.  Eye images and corresponding target locations are recorded automatically.  After the target motion is finished, calibration results are shown on the screen.  It is impossible to save calibration results to a file and reuse it later.

### for Offline tracker

*Command*

```
python -m rsgt.app.offline_calibration
```

A movie for calibration must be recorded in advance.  In addition, it is important to record when and where a target that participants look at is presented on the screen.  To start calibration, execute the above command and open the movie file.  Then, input when and where the target was presented and run calibration.
