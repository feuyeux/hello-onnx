package org.feuyeux.hello.ort;

import ai.onnxruntime.*;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.File;
import java.net.URL;
import java.util.Map;

@SpringBootTest
@Slf4j
public class HelloOnnxApplicationTests {
    // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/squeezenet/model.onnx
    private static final String modelName = "model.onnx";

    @Test
    public void textLoadOnnx() {
        var env = OrtEnvironment.getEnvironment();
        log.info("Hello onnxruntime, version:{}", env.getVersion());
        ClassLoader classLoader = getClass().getClassLoader();
        URL resource = classLoader.getResource(modelName);
        if (resource != null) {
            File file = new File(resource.getFile());
            String modelPath = file.getAbsolutePath();
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);
                options.setMemoryPatternOptimization(true);
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#gpu_mem_limit
                String arenaInBytes="4096000";
                options.addConfigEntry("gpu_mem_limit",arenaInBytes);
                Map<String, String> configEntries = options.getConfigEntries();
                log.info("model config entries:");
                configEntries.forEach((k, v) -> {
                    log.info("{}:{}", k, v);
                });
                try (OrtSession session = env.createSession(modelPath, options)) {
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
                }
            } catch (OrtException e) {
                log.error("", e);
            }
        } else {
            log.error("No onnx model found");
        }
    }
}
