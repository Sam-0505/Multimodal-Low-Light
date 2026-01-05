<div align="center">
<h1>Real-Time Multimodal Low-light Enhancement For Edge Devices</h1>
</div>
The goal is to adapt "Wakeup-Darkness" multimodal enhancement model to achieve the best possible quality-to-speed ratio for real-time XR deployment. The objective was to optimize the computational bottlenecks (segmentation & depth) while retaining the core fusion logic and removed the grounding dino to use automatic generation of FastSAM of create full segmentation maps.

## Steps to run low-light enhanced real-time webcam video:

1) `pip install -r requirements.txt`
2) `cd realtime-enhancer`
3) `python server.py`
4) `Select the models you want to use`
5) `Set the brightness threshold above which low-light enhancement should work`
6) `Click on Start Camera & Compare Models`
7) `Click on Stop Camera to close webcam and reset values`

## Results
<img width="1402" height="757" alt="image" src="https://github.com/user-attachments/assets/7a29549f-313e-49cb-8d48-0734a3f7deb0" />
<img width="1414" height="778" alt="image" src="https://github.com/user-attachments/assets/725b2f9e-25ce-45fc-a97d-8e7760ba5655" />



