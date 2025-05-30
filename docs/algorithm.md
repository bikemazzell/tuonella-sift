Complete Algorithm
1. Initialization ✅ COMPLETED

    ✅ Query System Resources:
        ✅ Determine:
            ✅ Total and available GPU memory.
            ✅ Total and available system RAM.
        ✅ Parse user-specified memory limits:
            ✅ GPU_limit = min(available_gpu_memory, user_gpu_limit)
            ✅ RAM_limit = min(available_ram, user_ram_limit)

    ✅ Set Chunk Sizes:
        ✅ Calculate chunk sizes for processing:
            ✅ RAM Buffer Size: Allocate ~90% of the RAM_limit for buffering file chunks.
            ✅ GPU Chunk Size: Allocate ~90% of the GPU_limit for processing.

    ✅ Preallocate Buffers:
        ✅ Allocate a RAM buffer to hold file chunks.
        ✅ Allocate a GPU buffer to process chunks.

2. Processing Input CSV Files
2.1 Stream Files and Pre-Validate Lines ✅ COMPLETED

    ✅ For Each CSV in the Input Folder:
        ✅ Open the CSV file in streaming mode (read one line at a time).

    ✅ Line-by-Line Pre-Validation (CPU):
        ✅ For each line in the CSV:
            ✅ Skip the Line If:
                ✅ It contains no printable characters (e.g., binary, whitespace only).
                ✅ It does not contain at least one delimiter (e.g., comma, semicolon, tab, or pipe).
                ✅ It does not contain a valid username (email or printable characters based on config).
            ✅ If the line passes all checks, store it in the RAM buffer.

    ✅ Write Filtered Lines to a Temporary File:
        ✅ Once the RAM buffer is full or the file has been fully read:
            ✅ Write the valid lines to a temporary file.
        ✅ Repeat until all lines from the CSV are processed.

2.2 Transfer Validated Data to GPU ✅ COMPLETED

    ✅ For Each Temporary File Generated:
        ✅ Read the file in chunks of size RAM Buffer Size.
        ✅ Transfer chunks to the GPU buffer for parallel processing.

    ✅ GPU Processing (Parallelized):
        ✅ Use CUDA kernels to process the lines in parallel:
            ✅ Normalize Email Field:
                ✅ Locate the email field using regex and convert it to lowercase.
            ✅ Normalize URL Field:
                ✅ Locate the URL field using regex and:
                    ✅ If it starts with android://, strip everything except tdl.domain.name.
                    ✅ If it starts with http:// or https://, strip everything except tdl.domain.name.
                    ✅ If no protocol is specified, strip everything except tdl.domain.name.
                ✅ Remove trailing slashes or fragments after /.
            ✅ Parse the line into a Record struct with fields: username, password, normalized_url.
        ✅ Pass processed records back to the CPU for deduplication.

2.3 Deduplication and Hash Map Storage ✅ COMPLETED

    ✅ Insert Records into a Hash Map (CPU):
        ✅ For each record returned from the GPU:
            ✅ Use a composite key combining core fields: normalized_email|password|normalized_url.
            ✅ Records are considered duplicates if core fields (email, password, URL) are identical.
            ✅ Keep ALL records with the same email if they have ANY differences in core fields:
                ✅ Different passwords: user@email.com,123,site.com vs user@email.com,456,site.com (KEEP BOTH)
                ✅ Different URLs: user@email.com,123,first.com vs user@email.com,123,second.com (KEEP BOTH)
            ✅ If core fields are identical but field counts differ:
                ✅ Keep the record with MORE fields (more complete data)
                ✅ Example: user@email.com,pass,site.com vs user@email.com,pass,site.com,extra → Keep the longer one
            ✅ If the composite key already exists (exact duplicate):
                ✅ Compare the completeness of the existing record and the new record.
                ✅ Retain the record with the higher completeness score:
                    ✅ Completeness Scoring: More fields populated, greater character count, etc.

    ✅ Repeat for All Temporary Files:
        ✅ Process all temporary files generated in step 2.1, one at a time.

3. Final Output ✅ COMPLETED

    ✅ Write Final Records to a Temporary File:
        ✅ Once all temporary files have been processed and deduplicated, write the hash map contents to a final temporary file.
        ✅ Use the output format: username,password,normalized_url.

    ✅ Output the Final File:
        ✅ Write the final temporary file to the user-specified output location.
        ✅ Cleanup temporary files after successful processing.

4. Memory Management ✅ COMPLETED

    ✅ Dynamic Resource Allocation:
        ✅ Continuously monitor GPU and RAM usage.
        ✅ Dynamically adjust chunk sizes if memory usage approaches the user-specified limits:
            ✅ Increase chunk size if memory usage is low.
            ✅ Decrease chunk size if memory usage is high.

    ✅ Release Resources:
        ✅ Free GPU and RAM buffers after processing each chunk to avoid memory leaks.

5. Error Handling ✅ COMPLETED

    ✅ Handle Corrupted or Invalid Lines:
        ✅ Log invalid or skipped lines to a separate file for later review.

    ✅ Graceful Recovery:
        ✅ If memory limits are exceeded or an error occurs:
            ✅ Split the current chunk into smaller sub-chunks and retry.
            ✅ Ensure no data is lost by resuming from the last processed file or chunk.

6. Optimization

    ✅ Overlap File I/O and GPU Processing:
        ✅ While the GPU processes one chunk, the CPU loads and filters the next chunk into RAM.
        ✅ Use double buffering to overlap these operations.

    ✅ Adaptive Chunk Sizing:
        ✅ Dynamically adjust chunk sizes based on:
            ✅ Observed processing speed.
            ✅ Resource availability (GPU and RAM usage).

    ✅ Batch Processed Data:
        ✅ Batch multiple processed records into a single write operation to reduce I/O overhead.

    ✅ Streaming and Parallel Processing:
        ✅ Implement parallel file processing with thread pools.
        ✅ Add streaming optimizations for large file handling.
        ✅ Create parallel I/O operations for multiple files.

7. Username Validation Enhancement ✅ COMPLETED

    ✅ Configurable Username Validation:
        ✅ Added `email_username_only` boolean flag to [deduplication] config section
        ✅ When `email_username_only = true`: Only email addresses are accepted as usernames (original behavior)
        ✅ When `email_username_only = false`: Any printable character string is accepted as username

    ✅ Printable Username Validation:
        ✅ Validates usernames using printable ASCII characters (0x21-0x7E)
        ✅ Excludes URL-like strings (starting with http://, https://, android://, etc.)
        ✅ Excludes empty or whitespace-only fields
        ✅ Excludes very long strings that appear to be passwords (heuristic)

    ✅ Field Detection Enhancement:
        ✅ Updated field detection logic to work with both email and printable username modes
        ✅ Maintains backward compatibility with existing email-only configurations
        ✅ Comprehensive test coverage for both validation modes
