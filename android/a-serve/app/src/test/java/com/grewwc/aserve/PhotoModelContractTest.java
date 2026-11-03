package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public final class PhotoModelContractTest {
    private static final int HIDDEN = 768;
    private static final int EMBEDDING = 512;

    @Test
    public void oversizedPartialArchiveIsResetBeforeRetry() throws Exception {
        File partial = Files.createTempFile("photo-model", ".partial").toFile();
        try {
            Files.write(partial.toPath(), new byte[9]);
            assertEquals(0, PhotoModelManager.preparePartialForResume(partial, 8));
            assertEquals(0, partial.length());
        } finally {
            partial.delete();
        }
    }

    @Test
    public void pinnedModelsShareTheExpectedClipSpace() throws Exception {
        File assets = new File("build/generated/photo-models");
        File textPath = new File(assets, "clip_text.onnx");
        File visionPath = new File(assets, "clip_vision.onnx");
        assertTrue(textPath.isFile());
        assertTrue(visionPath.isFile());
        Map<String, Integer> vocab = loadVocab(new File(assets, "clip_vocab.txt"));
        float[] projection = readProjection(new File(assets, "clip_dense.safetensors"));

        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        try (OrtSession text = environment.createSession(textPath.getAbsolutePath());
             OrtSession vision = environment.createSession(visionPath.getAbsolutePath())) {
            assertTrue(text.getInputNames().contains("input_ids"));
            assertTrue(text.getInputNames().contains("attention_mask"));
            assertTrue(text.getOutputNames().contains("last_hidden_state"));
            assertTrue(vision.getInputNames().contains("pixel_values"));
            assertTrue(vision.getOutputNames().contains("image_embeds"));

            float[] image = visionEmbedding(environment, vision, solidRedPixels());
            float[] red = textEmbedding(environment, text, projection, vocab,
                    List.of("[CLS]", "a", "red", "square", "[SEP]"));
            float[] blue = textEmbedding(environment, text, projection, vocab,
                    List.of("[CLS]", "a", "blue", "square", "[SEP]"));
            assertEquals(EMBEDDING, image.length);
            assertTrue("A red image must rank the red caption above the blue caption",
                    dot(image, red) > dot(image, blue));
        }
    }

    private static float[] visionEmbedding(OrtEnvironment environment, OrtSession session,
                                           float[] pixels) throws Exception {
        try (OnnxTensor input = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels),
                new long[]{1, 3, 224, 224});
             OrtSession.Result result = session.run(Map.of("pixel_values", input))) {
            float[] embedding = ((float[][]) result.get("image_embeds").orElseThrow().getValue())[0].clone();
            normalize(embedding);
            return embedding;
        }
    }

    private static float[] textEmbedding(OrtEnvironment environment, OrtSession session,
                                         float[] projection, Map<String, Integer> vocab,
                                         List<String> tokens) throws Exception {
        long[][] ids = new long[1][tokens.size()];
        long[][] mask = new long[1][tokens.size()];
        for (int index = 0; index < tokens.size(); index++) {
            Integer id = vocab.get(tokens.get(index));
            if (id == null) throw new IllegalStateException("Missing token " + tokens.get(index));
            ids[0][index] = id;
            mask[0][index] = 1;
        }
        try (OnnxTensor inputIds = OnnxTensor.createTensor(environment, ids);
             OnnxTensor attentionMask = OnnxTensor.createTensor(environment, mask)) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIds);
            inputs.put("attention_mask", attentionMask);
            try (OrtSession.Result result = session.run(inputs)) {
                float[][] hidden = ((float[][][]) result.get("last_hidden_state").orElseThrow().getValue())[0];
                assertEquals(HIDDEN, hidden[0].length);
                float[] pooled = new float[HIDDEN];
                for (float[] token : hidden) {
                    for (int column = 0; column < HIDDEN; column++) pooled[column] += token[column];
                }
                for (int column = 0; column < HIDDEN; column++) pooled[column] /= hidden.length;
                float[] output = new float[EMBEDDING];
                for (int row = 0; row < EMBEDDING; row++) {
                    for (int column = 0; column < HIDDEN; column++) {
                        output[row] += projection[row * HIDDEN + column] * pooled[column];
                    }
                }
                normalize(output);
                return output;
            }
        }
    }

    private static float[] solidRedPixels() {
        float[] pixels = new float[3 * 224 * 224];
        float red = (1.0f - 0.48145466f) / 0.26862954f;
        float green = (0.0f - 0.4578275f) / 0.26130258f;
        float blue = (0.0f - 0.40821073f) / 0.27577711f;
        for (int index = 0; index < 224 * 224; index++) {
            pixels[index] = red;
            pixels[224 * 224 + index] = green;
            pixels[2 * 224 * 224 + index] = blue;
        }
        return pixels;
    }

    private static Map<String, Integer> loadVocab(File file) throws Exception {
        Map<String, Integer> vocab = new HashMap<>();
        List<String> lines = Files.readAllLines(file.toPath());
        for (int index = 0; index < lines.size(); index++) vocab.put(lines.get(index), index);
        return vocab;
    }

    private static float[] readProjection(File file) throws Exception {
        byte[] bytes = Files.readAllBytes(file.toPath());
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int headerLength = Math.toIntExact(buffer.getLong());
        buffer.position(8 + headerLength);
        FloatBuffer values = buffer.slice().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] projection = new float[EMBEDDING * HIDDEN];
        values.get(projection);
        return projection;
    }

    private static float dot(float[] left, float[] right) {
        float value = 0.0f;
        for (int index = 0; index < left.length; index++) value += left[index] * right[index];
        return value;
    }

    private static void normalize(float[] values) {
        double square = 0.0;
        for (float value : values) square += value * value;
        float length = (float) Math.sqrt(square);
        for (int index = 0; index < values.length; index++) values[index] /= length;
    }
}
