# Face_tracking_turret
## Overview
This is a project created for the course Embedded Systems of the University of Western Macedonia. The system is based on the PoseNet deep learning model and is designed to track human faces that appear on the frame and follow the closest one autonomously. The turret can rotate continuously on its vertical Z-axis and tilt up to 75° on the horizontal Y-axis, allowing precise targeting. By moving the camera both on Z and Y, it creates a closed-loop control system that eliminates the need for encoders in the motors. It also corrects rotational errors using a PID conroller. Distance estimation for face switching is done by measuring face size in video frames rather than using depth sensing, which, while potentially less reliable, proved effective in testing.
<br><br><br>
There are expicit instructions for the hardware and software in the files attached.

![turret_picture_front_close](https://github.com/user-attachments/assets/7b8c925d-7587-40a1-87d1-de2f2f0a5785)
![turret_picture_side](https://github.com/user-attachments/assets/2d707326-05c0-4325-af4a-f6c57051c0c8)
![turret_working_video](https://github.com/user-attachments/assets/6dbc0e81-f014-4ed1-896c-df14045e38a9)


