/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.cloud.bigquery;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.google.api.services.bigquery.model.TableCell;
import com.google.api.services.bigquery.model.TableRow;
import com.google.common.collect.ImmutableList;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ReadChannel;
import org.apache.arrow.vector.ipc.WriteChannel;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.ipc.message.MessageSerializer;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.junit.jupiter.api.Test;

public class ITArrowVsJsonBenchmarkTest {

  private static final int NUM_ROWS = 50_000;

  @Test
  public void runArrowVsJsonBenchmark() throws IOException {
    System.out.println("=================================================");
    System.out.println("   BIGQUERY IN-MEMORY BENCHMARK: ARROW VS JSON    ");
    System.out.println("   Target Row Count: " + NUM_ROWS);
    System.out.println("=================================================");

    // 1. Define Schema
    Schema schema =
        Schema.of(
            Field.of("id", StandardSQLTypeName.INT64),
            Field.of("name", StandardSQLTypeName.STRING),
            Field.of("score", StandardSQLTypeName.FLOAT64),
            Field.of("is_active", StandardSQLTypeName.BOOL));

    // 2. Prepare Synthetic Data in JSON Format (List<TableRow>)
    List<TableRow> jsonRows = createJsonRows(NUM_ROWS);

    // 3. Prepare Synthetic Data in Base64-Encoded Arrow String Format (matches REST jobs.query Page 1 payload)
    byte[] rawArrowBytes = createArrowBatchBytes(NUM_ROWS);
    String base64ArrowString = Base64.getEncoder().encodeToString(rawArrowBytes);

    // -------------------------------------------------------------------------
    // BENCHMARK 1: JSON Parsing (TableRow -> FieldValueList)
    // -------------------------------------------------------------------------
    System.gc();
    long startMemoryJson = getUsedMemory();
    long startTimeJson = System.nanoTime();

    List<FieldValueList> jsonResultRows = parseJsonRows(jsonRows, schema);

    long elapsedJsonMs = (System.nanoTime() - startTimeJson) / 1_000_000;
    long memoryUsedJsonMb = (getUsedMemory() - startMemoryJson) / (1024 * 1024);

    System.out.println("\n--- [JSON Parsing Results] ---");
    System.out.println("Time Taken      : " + elapsedJsonMs + " ms");
    System.out.println("Rows Converted  : " + jsonResultRows.size());
    System.out.println("Approx Heap Used: " + Math.max(0, memoryUsedJsonMb) + " MB");

    // -------------------------------------------------------------------------
    // BENCHMARK 2: Base64 Decoding + Arrow Vector Decoding (Base64 String -> byte[] -> FieldValueList)
    // -------------------------------------------------------------------------
    System.gc();
    long startMemoryArrow = getUsedMemory();
    long startTimeArrow = System.nanoTime();

    // Includes explicit Base64 decoding step as requested by reviewer
    byte[] decodedArrowBytes = Base64.getDecoder().decode(base64ArrowString);
    List<FieldValueList> arrowResultRows = parseArrowBytes(decodedArrowBytes, schema);

    long elapsedArrowMs = (System.nanoTime() - startTimeArrow) / 1_000_000;
    long memoryUsedArrowMb = (getUsedMemory() - startMemoryArrow) / (1024 * 1024);

    System.out.println("\n--- [Base64 + Arrow Vector Decoding Results] ---");
    System.out.println("Time Taken      : " + elapsedArrowMs + " ms");
    System.out.println("Rows Converted  : " + arrowResultRows.size());
    System.out.println("Approx Heap Used: " + Math.max(0, memoryUsedArrowMb) + " MB");

    // -------------------------------------------------------------------------
    // SPEEDUP SUMMARY
    // -------------------------------------------------------------------------
    double speedupRatio = (double) elapsedJsonMs / Math.max(1, elapsedArrowMs);
    System.out.println("\n=================================================");
    System.out.printf("   SPEEDUP RATIO (JSON time / Arrow time): %.2fx\n", speedupRatio);
    System.out.println("=================================================\n");

    // Assert parity
    assertEquals(jsonResultRows.size(), arrowResultRows.size());
    assertNotNull(jsonResultRows.get(0));
    assertNotNull(arrowResultRows.get(0));
  }

  private List<TableRow> createJsonRows(int count) {
    List<TableRow> rows = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      TableRow row = new TableRow();
      List<TableCell> cells = new ArrayList<>(4);
      cells.add(new TableCell().setV(String.valueOf(i)));
      cells.add(new TableCell().setV("User_" + i));
      cells.add(new TableCell().setV(String.valueOf(i * 1.5)));
      cells.add(new TableCell().setV(i % 2 == 0 ? "true" : "false"));
      row.setF(cells);
      rows.add(row);
    }
    return rows;
  }

  private List<FieldValueList> parseJsonRows(List<TableRow> tableRows, Schema schema) {
    FieldList fields = schema.getFields();
    List<FieldValueList> result = new ArrayList<>(tableRows.size());
    for (TableRow rowPb : tableRows) {
      result.add(FieldValueList.fromPb(rowPb.getF(), fields));
    }
    return result;
  }

  private byte[] createArrowBatchBytes(int count) throws IOException {
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      BigIntVector idVector = new BigIntVector("id", allocator);
      VarCharVector nameVector = new VarCharVector("name", allocator);
      Float8Vector scoreVector = new Float8Vector("score", allocator);
      BitVector activeVector = new BitVector("is_active", allocator);

      idVector.allocateNew(count);
      nameVector.allocateNew(count * 10, count);
      scoreVector.allocateNew(count);
      activeVector.allocateNew(count);

      for (int i = 0; i < count; i++) {
        idVector.set(i, i);
        nameVector.setSafe(i, ("User_" + i).getBytes());
        scoreVector.set(i, i * 1.5);
        activeVector.set(i, i % 2 == 0 ? 1 : 0);
      }

      idVector.setValueCount(count);
      nameVector.setValueCount(count);
      scoreVector.setValueCount(count);
      activeVector.setValueCount(count);

      List<FieldVector> vectors = ImmutableList.of(idVector, nameVector, scoreVector, activeVector);
      try (VectorSchemaRoot root = new VectorSchemaRoot(vectors)) {
        root.setRowCount(count);
        VectorUnloader unloader = new VectorUnloader(root);
        try (ArrowRecordBatch batch = unloader.getRecordBatch()) {
          ByteArrayOutputStream out = new ByteArrayOutputStream();
          WriteChannel channel = new WriteChannel(Channels.newChannel(out));
          MessageSerializer.serialize(channel, batch);
          return out.toByteArray();
        }
      }
    }
  }

  private List<FieldValueList> parseArrowBytes(byte[] arrowBytes, Schema schema)
      throws IOException {
    List<FieldValueList> rows = new ArrayList<>();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      ArrowRecordBatch batch =
          MessageSerializer.deserializeRecordBatch(
              new ReadChannel(new ByteArrayReadableSeekableByteChannel(arrowBytes)),
              allocator);

      BigIntVector idVector = new BigIntVector("id", allocator);
      VarCharVector nameVector = new VarCharVector("name", allocator);
      Float8Vector scoreVector = new Float8Vector("score", allocator);
      BitVector activeVector = new BitVector("is_active", allocator);
      List<FieldVector> vectors = ImmutableList.of(idVector, nameVector, scoreVector, activeVector);

      try (VectorSchemaRoot root = new VectorSchemaRoot(vectors)) {
        VectorLoader loader = new VectorLoader(root);
        loader.load(batch);
        batch.close();

        FieldList fields = schema.getFields();
        for (int i = 0; i < root.getRowCount(); i++) {
          List<FieldValue> rowValues = new ArrayList<>(fields.size());
          for (int col = 0; col < fields.size(); col++) {
            FieldVector curVec = root.getVector(fields.get(col).getName());
            rowValues.add(FieldValue.of(FieldValue.Attribute.PRIMITIVE, curVec.getObject(i)));
          }
          rows.add(FieldValueList.of(rowValues, fields));
        }
        root.clear();
      }
    }
    return rows;
  }

  private long getUsedMemory() {
    Runtime runtime = Runtime.getRuntime();
    return runtime.totalMemory() - runtime.freeMemory();
  }
}
