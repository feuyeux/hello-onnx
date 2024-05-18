package org.feuyeux.hello.ort;

import ai.onnxruntime.*;
import ai.onnxruntime.providers.CoreMLFlags;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import com.google.gson.Gson;
import lombok.extern.slf4j.Slf4j;
import org.feuyeux.hello.ort.pojo.Detection;
import org.feuyeux.hello.ort.session.HelloOrtSession;
import org.feuyeux.hello.ort.utils.ImageUtil;
import org.feuyeux.hello.ort.utils.ModelFactory;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.feuyeux.hello.ort.session.HelloOrtManager.getSessionOptions;
import static org.feuyeux.hello.ort.session.HelloOrtSession.getEnvironment;

// How to limit GPU memory usage in onnxruntime
// https://stackoverflow.com/questions/68497294/how-to-limit-gpu-memory-usage-in-onnxruntime
// How to close the session in onnxruntime
@SpringBootTest
@Slf4j
public class HelloOrtTests {
    // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/squeezenet/model.onnx
    private static final String modelName = "model.onnx";


    @Test
    public void testYolo() {
        ModelFactory modelFactory = new ModelFactory();
        HelloOrtSession inferenceSession;
        Gson gson;
        try {
            inferenceSession = modelFactory.getModel("./model.properties");
            gson = new Gson();
            String[] imageNames = new String[]{"Laptop and Mouse.png", "dog.jpg", "dog.png", "kite.jpg"};
            for (String imageName : imageNames) {
                Mat img = Imgcodecs.imread("src/main/resources/" + imageName, Imgcodecs.IMREAD_COLOR);
                List<Detection> detectionList = inferenceSession.run(img);
                ImageUtil.drawPredictions(img, detectionList);
                log.info("detectionList:{}", gson.toJson(detectionList));
                Imgcodecs.imwrite("/tmp/prediction-" + imageName, img);
            }
        } catch (OrtException | IOException e) {
            log.debug("", e);
            System.exit(1);
        }
    }

    @Test
    public void textLoadOnnx() throws InterruptedException {
        OrtEnvironment env = getEnvironment();
        log.info("Hello onnxruntime, version:{}", env.getVersion());
        ClassLoader classLoader = getClass().getClassLoader();
        URL resource = classLoader.getResource(modelName);
        if (resource != null) {
            File file = new File(resource.getFile());
            String  modelPath = file.getAbsolutePath();
            try (OrtSession session = env.createSession(modelPath, getSessionOptions())) {
                log.info("model path:{}", modelPath);
                log.info("model num inputs:{}", session.getNumInputs());
                Map<String, NodeInfo> inputInfoList = session.getInputInfo();
                NodeInfo input = inputInfoList.get("data_0");
                TensorInfo inputInfo = (TensorInfo) input.getInfo();
                int[] expectedInputDimensions = new int[]{1, 3, 224, 224};
                for (int i = 0; i < expectedInputDimensions.length; i++) {
                    log.info("{}:{}", expectedInputDimensions[i], inputInfo.getShape()[i]);
                }
                Map<String, NodeInfo> outputInfoList = session.getOutputInfo();
                NodeInfo output = outputInfoList.get("softmaxout_1");
                TensorInfo outputInfo = (TensorInfo) output.getInfo();
                int[] expectedOutputDimensions = new int[]{1, 1000, 1, 1};
                for (int i = 0; i < expectedOutputDimensions.length; i++) {
                    log.info("{}:{}", expectedOutputDimensions[i], outputInfo.getShape()[i]);
                }
            } catch (OrtException e) {
                log.error("", e);
            }
        } else {
            log.error("No onnx model found");
        }
        TimeUnit.SECONDS.sleep(5);
    }
}
