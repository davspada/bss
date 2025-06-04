# bss
Blind Source Separation project

### **U-Net Architecture**
This U-Net model follows an encoder-decoder structure with skip connections, well suited for this type of tasks.  

#### Architecture

1. **Input & Reshape**: The `(32,32)` grayscale input is reshaped to `(32,32,1)` for compatibility with `Conv2D`.  
2. **Encoder**: Three convolutional blocks, each with:  
   - Two `Conv2D` layers (`ReLU` activation)  
   - `BatchNormalization` (after activation)  
   - `MaxPooling2D` (downsampling)  
   - Filters start at `64`, doubling at each step.  
3. **Bottleneck**: Two `Conv2D` layers (`ReLU` + `BatchNormalization`), extracting deep features.  
4. **Decoder**: Mirrors the encoder using `Conv2DTranspose` (upsampling) + skip connections for detail recovery.  
5. **Output**: A shared `Conv2D(8)` layer branches into two `1x1 Conv2D` layers (`sigmoid` activation), reshaped back to `(32,32)`.  

#### Key Design Choices  
- **Reshape**: Ensures compatibility with `Conv2D` and maintains original dimensions in output, without the need of a "squeeze" later on.  
- **Activation in `Conv2D`**: By adding the  activation function in the Conv2D layer, I avoid to use an extra layer for the activation itself.  
- **Skip Connections**: Prevent information loss, improving separation performance.  
