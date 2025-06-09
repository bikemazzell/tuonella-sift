# Tuonella Sift - Actual Implementation Algorithm

## Overview
Tuonella Sift uses a **HashMap-based deduplication** approach rather than external sorting. This provides better performance for deduplication while avoiding the complexity and I/O overhead of merge sort.

## 1. Initialization ✅ COMPLETED

    ✅ Load Configuration:
        ✅ Parse JSON configuration file (cpu.config.json or cuda.config.json)
        ✅ Set up memory limits, batch sizes, and processing parameters
        ✅ Configure username validation mode (email-only vs printable)
    
    ✅ Initialize Memory Manager:
        ✅ Query available system RAM
        ✅ Calculate working memory based on configured percentage
        ✅ Set up memory monitoring and pressure detection
        ✅ Initialize checkpoint handler for resume capability
    
    ✅ CUDA Setup (if enabled):
        ✅ Check CUDA availability and GPU memory
        ✅ Initialize CUDA context and memory pools
        ✅ Configure GPU batch sizes based on available memory

## 2. File Discovery and Pre-Processing ✅ COMPLETED

### 2.1 File Discovery
    ✅ Scan input directory recursively for CSV files
    ✅ Sort files for deterministic processing order
    ✅ Group small files into batches for efficient processing
    ✅ Track total size for progress estimation

### 2.2 Streaming Pre-Validation ✅ COMPLETED
    ✅ For Each CSV File:
        ✅ Open file with buffered reader (8KB buffer)
        ✅ Auto-detect delimiter (comma, semicolon, tab, or pipe)
        ✅ Detect field positions using intelligent sampling
    
    ✅ Line-by-Line Validation:
        ✅ Parse each line with detected delimiter
        ✅ Validate username field:
            ✅ Email mode: Must be valid email address
            ✅ Printable mode: Must contain printable ASCII chars
            ✅ Reject URL-like strings in username field
        ✅ Validate password field (must exist)
        ✅ Validate URL field (optional normalization)
        ✅ Skip lines with fewer than 3 fields
    
    ✅ Batch Writing:
        ✅ Accumulate valid records in memory buffer
        ✅ Write to temporary file when buffer reaches threshold
        ✅ Track validation errors in separate log file

## 3. HashMap-Based Deduplication ✅ COMPLETED

### 3.1 In-Memory Deduplication
    ✅ Initialize HashMap<String, Record> for storing unique records
    ✅ Process validated records in batches:
        ✅ For each record:
            ✅ Normalize username (lowercase)
            ✅ Normalize URL:
                ✅ Strip protocols (http://, https://, android://, etc.)
                ✅ Remove www. prefix
                ✅ Keep only domain and path
                ✅ Remove trailing slashes
            ✅ Create composite key: "username|password|normalized_url"
            ✅ Check if key exists in HashMap:
                ✅ If new: Add to HashMap
                ✅ If exists: Compare completeness scores
                ✅ Keep record with higher score (more fields/data)
    
### 3.2 Memory Management
    ✅ Monitor HashMap size against max_memory_records limit
    ✅ When approaching limit:
        ✅ Flush HashMap contents to output file
        ✅ Clear HashMap and continue processing
        ✅ Update checkpoint for resume capability
    
### 3.3 GPU Acceleration (Optional)
    ✅ If CUDA enabled and beneficial:
        ✅ Batch records for GPU processing
        ✅ Use CUDA kernels for parallel normalization
        ✅ Transfer normalized records back to CPU
        ✅ Continue with HashMap deduplication on CPU

## 4. Completeness Scoring ✅ COMPLETED

### 4.1 Duplicate Resolution Logic
    ✅ When duplicate key found, calculate completeness scores:
        ✅ Base score = number of fields
        ✅ Bonus score = 0.1 * total character count
        ✅ Total score = base + bonus
    
    ✅ Keep record with highest score:
        ✅ More fields = higher priority
        ✅ Longer field values = tiebreaker
        ✅ Preserves most complete information

### 4.2 Deduplication Rules
    ✅ Same user + password + URL = Duplicate (keep best)
    ✅ Same user, different password = Different records (keep both)
    ✅ Same user, different URL = Different records (keep both)
    ✅ Different users = Different records (keep both)

## 5. Output Generation ✅ COMPLETED

### 5.1 Final Output Writing
    ✅ Create output file with standardized format:
        ✅ Header: username,password,url[,additional_fields...]
        ✅ Preserve all original fields beyond core three
        ✅ Write in batches for I/O efficiency
    
    ✅ Output Organization:
        ✅ Single output file (no sorting applied)
        ✅ Records in HashMap iteration order
        ✅ Configurable output size limits (future feature)

### 5.2 Cleanup
    ✅ Remove all temporary files
    ✅ Delete checkpoint files after successful completion
    ✅ Log final statistics

## 6. Memory and Resource Management ✅ COMPLETED

### 6.1 Memory Monitoring
    ✅ Track system memory usage in real-time
    ✅ Detect memory pressure conditions
    ✅ Adjust batch sizes dynamically:
        ✅ Reduce when memory pressure detected
        ✅ Increase when memory available
    
### 6.2 Resource Limits
    ✅ Respect configured memory percentage
    ✅ Flush HashMap before reaching limits
    ✅ GPU memory management (if CUDA enabled)
    ✅ Temporary file size management

## 7. Error Handling and Recovery ✅ COMPLETED

### 7.1 Validation Error Handling
    ✅ Log invalid records with reasons:
        ✅ No printable characters
        ✅ Missing delimiter
        ✅ Invalid username format
        ✅ Insufficient fields
    ✅ Write errors to validation_errors.log
    ✅ Continue processing without data loss

### 7.2 Checkpoint and Resume
    ✅ Save progress at regular intervals:
        ✅ Current file being processed
        ✅ Records processed count
        ✅ Current phase (pre-processing/deduplication)
    ✅ Resume from checkpoint after interruption:
        ✅ Skip already processed files
        ✅ Continue from last position
        ✅ Restore statistics and counters
    
### 7.3 Signal Handling
    ✅ Graceful shutdown on Ctrl+C
    ✅ Save current state before exit
    ✅ Allow resume from interruption point

## 8. Performance Optimizations ✅ COMPLETED

### 8.1 I/O Optimizations
    ✅ Buffered reading with 8KB buffers
    ✅ Batch writing to reduce syscalls
    ✅ Async I/O for parallel file processing
    ✅ Memory-mapped files for large datasets

### 8.2 Processing Optimizations
    ✅ Pre-compiled regex patterns
    ✅ SIMD string operations where available
    ✅ Parallel processing with thread pools
    ✅ Lock-free data structures where possible

### 8.3 Memory Optimizations
    ✅ Reusable buffer pools
    ✅ Efficient string handling
    ✅ Minimal allocations in hot paths
    ✅ Strategic HashMap sizing

### 8.4 CUDA Optimizations (Optional)
    ✅ Pinned memory for faster transfers
    ✅ Multiple CUDA streams for overlap
    ✅ Optimized kernel configurations
    ✅ Batch processing for GPU efficiency

## 9. Configuration and Features ✅ COMPLETED

### 9.1 Username Validation Modes
    ✅ Email-only mode (default):
        ✅ Strict email validation
        ✅ RFC-compliant email parsing
    ✅ Printable character mode:
        ✅ Accept any printable ASCII string
        ✅ Exclude URL-like patterns
        ✅ Heuristic password detection

### 9.2 Performance Monitoring
    ✅ Real-time progress tracking
    ✅ ETA calculations
    ✅ Throughput measurements
    ✅ Memory usage statistics

### 9.3 Configuration Options
    ✅ Memory limits (percentage-based)
    ✅ Batch sizes and chunk sizes
    ✅ Thread counts and parallelism
    ✅ CUDA enablement
    ✅ Verbosity levels
    ✅ Progress intervals

## 10. Key Differences from External Sort Algorithm

### Why HashMap Instead of External Sort?
    ✅ **Better for Deduplication**: Direct lookup is O(1) vs O(log n) for sorted data
    ✅ **Less I/O**: No intermediate sorted chunks to write/read
    ✅ **Simpler Implementation**: No complex k-way merge needed
    ✅ **Memory Efficient**: Only stores unique records in memory
    ✅ **Preserves Input Order**: No unnecessary reordering

### Trade-offs
    ❌ Output is not sorted (rarely needed for deduplicated data)
    ❌ Cannot handle datasets larger than available memory partitions
    ✅ Perfect for deduplication use case where uniqueness matters more than order

### The External Sort Module
    The `external_sort_dedup.rs` module exists but is not used in the main flow. It could be activated for:
    - Scenarios requiring sorted output
    - Datasets too large for HashMap approach
    - Future enhancement for ordered deduplication
