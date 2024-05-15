package org.feuyeux.hello.ort.pojo;

public record Detection(String label, float[] bbox, float confidence) {

}
