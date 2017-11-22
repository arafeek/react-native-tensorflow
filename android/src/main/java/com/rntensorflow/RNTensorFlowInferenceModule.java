
package com.rntensorflow;

import com.facebook.react.bridge.*;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.contrib.android.RunStats;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.io.File;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Environment;

import static com.rntensorflow.converter.ArrayConverter.*;

public class RNTensorFlowInferenceModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;
  private Map<String, TfContext> tfContexts = new HashMap<>();

  public RNTensorFlowInferenceModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "RNTensorFlowInference";
  }

  @Override
  public void onCatalystInstanceDestroy() {
    for (String id : tfContexts.keySet()) {
      TfContext tfContext = this.tfContexts.remove(id);
      if(tfContext != null) {
        tfContext.session.close();
      }
    }
  }

  @ReactMethod
  public void initTensorFlowInference(String id, String model, Promise promise) {
    try {
      loadNativeTf();
      TfContext context = createContext(model);
      tfContexts.put(id, context);

      RNTensorFlowGraphModule graphModule = reactContext.getNativeModule(RNTensorFlowGraphModule.class);
      graphModule.init(id, context.graph);

      promise.resolve(true);
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  private void loadNativeTf() {
    try {
      new RunStats();
    } catch (UnsatisfiedLinkError ule) {
      System.loadLibrary("tensorflow_inference");
    }
  }

  private TfContext createContext(String model) throws IOException {
    byte[] b = new ResourceManager(reactContext.getAssets()).loadResource(model);

    Graph graph = new Graph();
    graph.importGraphDef(b);
    Session session = new Session(graph);
    Session.Runner runner = session.runner();

    return new TfContext(session, runner, graph);
  }


  private float [] getImageData(String imagePath) {
    // https://stackoverflow.com/questions/16804404/create-a-bitmap-drawable-from-file-path

    // 1) Get a bitmap representation of the image
    int width = 500;
    int height = 200;
    int channels = 3; // RGB only

    float [] output = new float[width*height*channels];

//    File sd = Environment.getExternalStorageDirectory();
//    File image = new File(sd + imagePath, "my_image");
    BitmapFactory.Options bitmapOpts = new BitmapFactory.Options();
    // TODO: If this is null, throw an error
    Bitmap bitmap = BitmapFactory.decodeFile(imagePath, bitmapOpts);
    // need to make a copy as decoded bitmaps are readonly
    Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
    // 2) Resize the bitmap to 500 x 200 (or whatever the model expects)
    // TODO: make this usable for other image sizes
    mutableBitmap.reconfigure(width, height, Bitmap.Config.ARGB_8888);
    // 3) Convert this 500 x 200 x 4 image to a 500 x 200 x 3 pixel array
    int red, green, blue, pixel;
    int i = 0;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        pixel = mutableBitmap.getPixel(x, y);
        red = Color.red(pixel);
        blue = Color.blue(pixel);
        green = Color.green(pixel);

        // TODO: find a better way than manually casting these
        output[i] = (float) red;
        output[++i] = (float) blue;
        output[++i] = (float) green;
      }
    }

    return output;
  }

  @ReactMethod
  public void feedImage(String id, ReadableMap data, Promise promise) {
    try {
      String inputName = data.getString("name");
      long[] shape = data.hasKey("shape") ? readableArrayToLongArray(data.getArray("shape")) : new long[0];

      TfContext tfContext = tfContexts.get(id);
      if (tfContext != null) {
        String imagePath = data.getString("image");
        float[] srcData = getImageData(imagePath);
        tfContext.runner.feed(inputName, Tensor.create(shape, FloatBuffer.wrap(srcData)));
        promise.resolve(true);
      } else {
        promise.reject(new IllegalStateException("Could not find tfContext for id"));
      }
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  @ReactMethod
  public void feed(String id, ReadableMap data, Promise promise) {
    try {
      String inputName = data.getString("name");
      long[] shape = data.hasKey("shape") ? readableArrayToLongArray(data.getArray("shape")) : new long[0];

      DataType dtype = data.hasKey("dtype")
              ? DataType.valueOf(data.getString("dtype").toUpperCase())
              : DataType.DOUBLE;

      TfContext tfContext = tfContexts.get(id);
      if (tfContext != null) {
        if(dtype == DataType.DOUBLE) {
          double[] srcData = readableArrayToDoubleArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(shape, DoubleBuffer.wrap(srcData)));
        } else if(dtype == DataType.FLOAT) {
          float[] srcData = readableArrayToFloatArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(shape, FloatBuffer.wrap(srcData)));
        } else if(dtype == DataType.INT32) {
          int[] srcData = readableArrayToIntArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(shape, IntBuffer.wrap(srcData)));
        } else if(dtype == DataType.INT64) {
          double[] srcData = readableArrayToDoubleArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(shape, DoubleBuffer.wrap(srcData)));
        } else if(dtype == DataType.UINT8) {
          int[] srcData = readableArrayToIntArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(shape, IntBuffer.wrap(srcData)));
        } else if(dtype == DataType.BOOL) {
          byte[] srcData = readableArrayToByteBoolArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(dtype, shape, ByteBuffer.wrap(srcData)));
        } else if(dtype == DataType.STRING) {
          byte[] srcData = readableArrayToByteStringArray(data.getArray("data"));
          tfContext.runner.feed(inputName, Tensor.create(dtype, shape, ByteBuffer.wrap(srcData)));
        } else {
          promise.reject(new IllegalArgumentException("Data type is not supported"));
          return;
        }
        promise.resolve(true);
      } else {
        promise.reject(new IllegalStateException("Could not find tfContext for id"));
      }
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  @ReactMethod
  public void run(String id, ReadableArray outputNames, boolean enableStats, Promise promise) {
    try {
      TfContext tfContext = tfContexts.get(id);
      if(tfContext != null) {
        String[] outputNamesString = readableArrayToStringArray(outputNames);
        for (String outputName : outputNamesString) {
          tfContext.runner.fetch(outputName);
        }
        List<Tensor> tensors = tfContext.runner.run();

        tfContext.outputTensors.clear();
        for (int i = 0; i < outputNamesString.length; i++) {
          tfContext.outputTensors.put(outputNamesString[i], tensors.get(i));
        }

        promise.resolve(true);
      } else {
        promise.reject(new IllegalStateException("Could not find inference for id"));
      }
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  @ReactMethod
  public void fetch(String id, String outputName, Promise promise) {
    try {
      TfContext tfContext = tfContexts.get(id);

      Tensor tensor = tfContext.outputTensors.get(outputName);
      int numElements = tensor.numElements();

      if(tensor.dataType() == DataType.DOUBLE) {
        DoubleBuffer dst = DoubleBuffer.allocate(numElements);
        tensor.writeTo(dst);
        promise.resolve(doubleArrayToReadableArray(dst.array()));
      } else if(tensor.dataType() == DataType.FLOAT) {
        FloatBuffer dst = FloatBuffer.allocate(numElements);
        tensor.writeTo(dst);
        promise.resolve(floatArrayToReadableArray(dst.array()));
      } else if(tensor.dataType() == DataType.INT32) {
        IntBuffer dst = IntBuffer.allocate(numElements);
        tensor.writeTo(dst);
        promise.resolve(intArrayToReadableArray(dst.array()));
      } else if(tensor.dataType() == DataType.INT64) {
        DoubleBuffer dst = DoubleBuffer.allocate(numElements);
        tensor.writeTo(dst);
        promise.resolve(doubleArrayToReadableArray(dst.array()));
      } else if(tensor.dataType() == DataType.UINT8) {
        IntBuffer dst = IntBuffer.allocate(numElements);
        tensor.writeTo(dst);
        promise.resolve(intArrayToReadableArray(dst.array()));
      } else if(tensor.dataType() == DataType.BOOL) {
        ByteBuffer dst = ByteBuffer.allocate(numElements);
        tensor.writeTo(dst);
        promise.resolve(byteArrayToBoolReadableArray(dst.array()));
      } else {
        promise.reject(new IllegalArgumentException("Data type is not supported"));
      }
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  @ReactMethod
  public void close(String id, Promise promise) {
    try {
      TfContext tfContext = this.tfContexts.remove(id);
      if(tfContext != null) {
        tfContext.session.close();
        tfContext.outputTensors.clear();
        promise.resolve(true);
      } else {
        promise.reject(new IllegalStateException("Could not find inference for id"));
      }
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  private class TfContext {
    final Session session;
    final Session.Runner runner;
    final Graph graph;
    final Map<String, Tensor> outputTensors;

    TfContext(Session session, Session.Runner runner, Graph graph) {
      this.session = session;
      this.runner = runner;
      this.graph = graph;
      outputTensors = new HashMap<>();
    }
  }
}
