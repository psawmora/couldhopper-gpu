# couldhopper-gpu
Couldhopper Smpp Encoder modified to use GPU
Cloudhopper is the most popular SMPP java library. Cloudhopper uses netty for implementing the networking layer.

The decoder used by the cloudhopper is a part of the netty pipeline and instantiated per pipeline.
Cloudhoppe-GPU deligates the decoding to a GPU and tries to gain a higher throughput.

The project uses JNI for communicating with the C implementation. 
As the project moves on, the README would be updated accordingly.

LD_LIBRARY_PATH=/usr/local/cuda/lib64