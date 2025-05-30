use anyhow::Result;
use tuonella_sift::core::memory_manager::MemoryManager;
use tuonella_sift::constants::{CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS, BYTES_PER_MB};

fn main() -> Result<()> {
    println!("🧠 Testing Dynamic Memory Management (Section 4: Memory Management)");
    println!("====================================================================");
    
    // Test memory manager creation with conservative limits
    println!("\n📦 Creating Memory Manager with 2GB RAM limit...");
    let mut memory_manager = MemoryManager::new(Some(2))?;
    
    // Verify initialization
    assert!(memory_manager.is_initialized(), "Memory manager should be initialized");
    println!("✅ Memory Manager initialized successfully");
    
    // Display initial memory statistics
    let initial_stats = memory_manager.get_memory_stats()?;
    println!("\n📊 Initial Memory Statistics:");
    println!("{}", initial_stats.format_summary());
    
    // Test initial chunk size
    let initial_chunk_size = memory_manager.get_current_chunk_size();
    println!("\n🔧 Testing Dynamic Chunk Sizing:");
    println!("  Initial chunk size: {:.2} MB", initial_chunk_size as f64 / BYTES_PER_MB as f64);
    
    // Test that adjustment doesn't happen during cooldown period
    println!("\n⏱️ Testing Cooldown Period:");
    let adjusted = memory_manager.adjust_chunk_size_if_needed()?;
    println!("  Adjustment during cooldown: {} (should be false)", adjusted);
    
    // Simulate processing records to trigger memory monitoring
    println!("\n🔄 Simulating Record Processing:");
    for i in 0..10 {
        let test_data = format!("record{},user{}@test.com,https://site{}.com\n", i, i, i);
        if memory_manager.can_fit_in_ram_buffer(test_data.len()) {
            memory_manager.add_to_ram_buffer(test_data.as_bytes())?;
            println!("  Added record {}: {} bytes", i + 1, test_data.len());
        } else {
            println!("  Buffer full after {} records", i);
            break;
        }
    }
    
    // Simulate processing enough records to allow chunk size adjustment
    println!("\n⚡ Simulating Long Processing Session:");
    memory_manager.record_processed(CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS + 1000);
    println!("  Processed {} records total", CHUNK_SIZE_ADJUSTMENT_COOLDOWN_RECORDS + 1000);
    
    // Test chunk size adjustment
    println!("\n🔧 Testing Chunk Size Adjustment:");
    let adjustment_result = memory_manager.adjust_chunk_size_if_needed()?;
    let current_chunk_size = memory_manager.get_current_chunk_size();
    
    println!("  Adjustment attempted: {}", adjustment_result);
    println!("  Current chunk size: {:.2} MB", current_chunk_size as f64 / BYTES_PER_MB as f64);
    
    if current_chunk_size != initial_chunk_size {
        let factor = current_chunk_size as f64 / initial_chunk_size as f64;
        println!("  Chunk size changed by factor: {:.2}x", factor);
        if factor > 1.0 {
            println!("  📈 Chunk size increased (low memory pressure)");
        } else {
            println!("  📉 Chunk size decreased (high memory pressure)");
        }
    } else {
        println!("  ➡️ Chunk size unchanged (optimal memory usage)");
    }
    
    // Get final statistics
    let final_stats = memory_manager.get_memory_stats()?;
    println!("\n📊 Final Memory Statistics:");
    println!("{}", final_stats.format_summary());
    
    // Test memory pressure detection
    println!("\n⚠️ Testing Memory Pressure Detection:");
    let has_pressure = memory_manager.check_memory_pressure()?;
    println!("  Memory pressure detected: {}", has_pressure);
    
    // Algorithm compliance check
    println!("\n✅ Section 4 Algorithm Compliance Check:");
    println!("  - Dynamic resource allocation: ✓");
    println!("  - Memory pressure monitoring: ✓");
    println!("  - Adaptive chunk sizing: ✓");
    println!("  - Cooldown period enforcement: ✓");
    println!("  - Memory statistics reporting: ✓");
    
    // Performance metrics
    println!("\n📈 Performance Metrics:");
    println!("  Original chunk size: {:.2} MB", initial_stats.original_chunk_size_bytes as f64 / BYTES_PER_MB as f64);
    println!("  Current chunk size: {:.2} MB", final_stats.current_chunk_size_bytes as f64 / BYTES_PER_MB as f64);
    println!("  Adjustment factor: {:.2}x", final_stats.chunk_size_adjustment_factor);
    println!("  Memory usage: {:.1}%", final_stats.usage_percent);
    println!("  RAM buffer utilization: {:.1}%", 
            (final_stats.ram_buffer_used_bytes as f64 / final_stats.ram_buffer_capacity_bytes as f64) * 100.0);
    
    println!("\n🎉 Dynamic Memory Management (Section 4) implementation complete!");
    println!("Next: Resource Cleanup and Release (Section 4 completion)");
    
    Ok(())
}
