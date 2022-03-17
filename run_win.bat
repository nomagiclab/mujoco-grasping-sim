nmake
SET var=%cd%
cd ../../bin && mujoco-grasping-sim
cd %var%
