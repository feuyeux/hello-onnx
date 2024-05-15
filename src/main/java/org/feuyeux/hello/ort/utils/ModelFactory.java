package org.feuyeux.hello.ort.utils;

import ai.onnxruntime.OrtException;
import org.feuyeux.hello.ort.exception.NotImplementedException;
import org.feuyeux.hello.ort.session.HelloOrtSession;
import org.feuyeux.hello.ort.session.HelloOrtSessionV5;
import org.feuyeux.hello.ort.session.HelloOrtSessionV8;
import org.springframework.util.ResourceUtils;

import java.io.File;
import java.io.IOException;
import java.util.Properties;

public class ModelFactory {


    public HelloOrtSession getModel(String propertiesFilePath) throws IOException, OrtException, NotImplementedException {

        ConfigReader configReader = new ConfigReader();
        Properties properties = configReader.readProperties(propertiesFilePath);

        String modelName = properties.getProperty("modelName");
        File file = ResourceUtils.getFile("classpath:" + properties.getProperty("modelPath"));
        File file2 = ResourceUtils.getFile("classpath:coco.names");
        String modelPath =  String.valueOf((file));
        String labelPath =  String.valueOf((file2));
        float confThreshold = Float.parseFloat(properties.getProperty("confThreshold"));
        float nmsThreshold = Float.parseFloat(properties.getProperty("nmsThreshold"));
        int gpuDeviceId = Integer.parseInt(properties.getProperty("gpuDeviceId"));

        if ("yolov5".equalsIgnoreCase(modelName)) {
            return new HelloOrtSessionV5(modelPath, labelPath, confThreshold, nmsThreshold, gpuDeviceId);
        }
        else if ("yolov8".equalsIgnoreCase(modelName)) {
            return new HelloOrtSessionV8(modelPath, labelPath, confThreshold, nmsThreshold, gpuDeviceId);
        }
        else {
            throw new NotImplementedException();
        }

    }
}
