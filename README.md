## Fluxel

I wanted to learn training diffusion models by implementing them with cooperative vectors and slang autodiff!

I'm starting with simple MLP-based diffusion models and gradually add more, maybe image generation models based on convolution, or even (patched) attention layers? (Hopefully I can achieve this...)

This is just a pet project of mine and I'm not familiar with the mathematical foundations of diffusion models so it should progress really slowly......

### Dependencies

This project should start with RTXNS which offers basic construction blocks, like linear layer wrappers and optimizers. It will also heavily rely on the autodiff features of Slang.