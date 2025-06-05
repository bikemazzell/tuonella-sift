use anyhow::Result;
use clap::Parser;
use std::time::{Duration, Instant};
use std::thread;
use sysinfo::{System, Pid};
use tuonella_sift::config::Config;

#[derive(Parser)]
#[command(name = "resource-monitor")]
#[command(about = "🔍 Resource Monitor: Real-time system resource monitoring for Tuonella Sift")]
#[command(version)]
struct Args {
    #[arg(short, long, default_value = "config.json", help = "Configuration file")]
    config: std::path::PathBuf,

    #[arg(short, long, default_value = "5", help = "Monitoring interval in seconds")]
    interval: u64,

    #[arg(short, long, help = "Monitor specific process ID")]
    pid: Option<u32>,

    #[arg(long, help = "Show GPU information")]
    gpu: bool,

    #[arg(long, help = "Continuous monitoring (Ctrl+C to stop)")]
    continuous: bool,

    #[arg(long, help = "Export monitoring data to CSV file")]
    export: Option<std::path::PathBuf>,
}

#[derive(Debug)]
struct ResourceSnapshot {
    timestamp: Instant,
    total_ram_gb: f64,
    available_ram_gb: f64,
    used_ram_gb: f64,
    ram_usage_percent: f64,
    process_ram_mb: f64,
    cpu_usage_percent: f64,
    #[cfg(feature = "cuda")]
    gpu_memory_used_mb: f64,
    #[cfg(feature = "cuda")]
    gpu_memory_total_mb: f64,
    #[cfg(feature = "cuda")]
    gpu_usage_percent: f64,
}

impl ResourceSnapshot {
    fn new(system: &mut System, target_pid: Option<Pid>) -> Result<Self> {
        system.refresh_all();

        let total_ram_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_ram_gb = system.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let used_ram_gb = total_ram_gb - available_ram_gb;
        let ram_usage_percent = (used_ram_gb / total_ram_gb) * 100.0;

        let (process_ram_mb, cpu_usage_percent) = if let Some(pid) = target_pid {
            if let Some(process) = system.process(pid) {
                let ram_mb = process.memory() as f64 / 1024.0; // Convert KB to MB
                let cpu_percent = process.cpu_usage() as f64;
                (ram_mb, cpu_percent)
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        #[cfg(feature = "cuda")]
        let (gpu_memory_used_mb, gpu_memory_total_mb, gpu_usage_percent) = 
            Self::get_gpu_info().unwrap_or((0.0, 0.0, 0.0));

        Ok(ResourceSnapshot {
            timestamp: Instant::now(),
            total_ram_gb,
            available_ram_gb,
            used_ram_gb,
            ram_usage_percent,
            process_ram_mb,
            cpu_usage_percent,
            #[cfg(feature = "cuda")]
            gpu_memory_used_mb,
            #[cfg(feature = "cuda")]
            gpu_memory_total_mb,
            #[cfg(feature = "cuda")]
            gpu_usage_percent,
        })
    }

    #[cfg(feature = "cuda")]
    fn get_gpu_info() -> Result<(f64, f64, f64)> {
        // Try to use nvidia-smi as fallback since cudarc might not be available in this context
        use std::process::Command;

        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = output_str.trim().split(',').collect();
                if parts.len() >= 3 {
                    let used_mb = parts[0].trim().parse::<f64>().unwrap_or(0.0);
                    let total_mb = parts[1].trim().parse::<f64>().unwrap_or(0.0);
                    let util_percent = parts[2].trim().parse::<f64>().unwrap_or(0.0);
                    return Ok((used_mb, total_mb, util_percent));
                }
            }
            _ => {}
        }

        // Fallback to cudarc if nvidia-smi fails
        use cudarc::driver::result;
        match result::mem_get_info() {
            Ok((free_memory, total_memory)) => {
                let used_memory = total_memory - free_memory;
                let used_mb = used_memory as f64 / (1024.0 * 1024.0);
                let total_mb = total_memory as f64 / (1024.0 * 1024.0);
                let usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;
                Ok((used_mb, total_mb, usage_percent))
            }
            Err(_) => Ok((0.0, 0.0, 0.0)),
        }
    }

    fn format_summary(&self) -> String {
        let mut summary = format!(
            "📊 System Resources at {:?}\n\
             🧠 RAM: {:.2} GB used / {:.2} GB total ({:.1}%)\n\
             💾 Available RAM: {:.2} GB\n",
            self.timestamp,
            self.used_ram_gb,
            self.total_ram_gb,
            self.ram_usage_percent,
            self.available_ram_gb
        );

        if self.process_ram_mb > 0.0 {
            summary.push_str(&format!(
                "🔍 Process RAM: {:.2} MB\n\
                 ⚡ Process CPU: {:.1}%\n",
                self.process_ram_mb,
                self.cpu_usage_percent
            ));
        }

        #[cfg(feature = "cuda")]
        {
            if self.gpu_memory_total_mb > 0.0 {
                summary.push_str(&format!(
                    "🖥️  GPU Memory: {:.2} MB used / {:.2} MB total ({:.1}%)\n",
                    self.gpu_memory_used_mb,
                    self.gpu_memory_total_mb,
                    self.gpu_usage_percent
                ));
            } else {
                summary.push_str("🖥️  GPU: Not available or not accessible\n");
            }
        }

        summary
    }

    fn to_csv_header() -> String {
        #[cfg(feature = "cuda")]
        return "timestamp,total_ram_gb,available_ram_gb,used_ram_gb,ram_usage_percent,process_ram_mb,cpu_usage_percent,gpu_memory_used_mb,gpu_memory_total_mb,gpu_usage_percent".to_string();
        
        #[cfg(not(feature = "cuda"))]
        return "timestamp,total_ram_gb,available_ram_gb,used_ram_gb,ram_usage_percent,process_ram_mb,cpu_usage_percent".to_string();
    }

    fn to_csv_row(&self) -> String {
        let timestamp_secs = self.timestamp.elapsed().as_secs_f64();
        
        #[cfg(feature = "cuda")]
        return format!(
            "{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            timestamp_secs,
            self.total_ram_gb,
            self.available_ram_gb,
            self.used_ram_gb,
            self.ram_usage_percent,
            self.process_ram_mb,
            self.cpu_usage_percent,
            self.gpu_memory_used_mb,
            self.gpu_memory_total_mb,
            self.gpu_usage_percent
        );
        
        #[cfg(not(feature = "cuda"))]
        return format!(
            "{:.3},{:.3},{:.3},{:.3},{:.2},{:.2},{:.2}",
            timestamp_secs,
            self.total_ram_gb,
            self.available_ram_gb,
            self.used_ram_gb,
            self.ram_usage_percent,
            self.process_ram_mb,
            self.cpu_usage_percent
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("🔍 Tuonella Sift Resource Monitor");
    println!("==================================");

    // Load configuration to show current settings
    let config = Config::load(&args.config).await?;
    println!("📚 Configuration loaded from {}", args.config.display());
    println!("🧠 Max RAM configured: {}%", config.memory.memory_usage_percent);
    println!("📦 Chunk size: {} MB", config.processing.chunk_size_mb);
    println!("🔢 Record chunk size: {}", config.processing.record_chunk_size);
    println!("💾 Max memory records: {}", config.processing.max_memory_records);

    #[cfg(feature = "cuda")]
    if config.processing.enable_cuda {
        println!("🚀 CUDA enabled - GPU memory usage: {}%", config.cuda.gpu_memory_usage_percent);
        println!("📊 GPU batch sizes: small={}, medium={}, large={}, xlarge={}", 
                config.cuda.batch_sizes.small,
                config.cuda.batch_sizes.medium,
                config.cuda.batch_sizes.large,
                config.cuda.batch_sizes.xlarge);
    }

    let mut system = System::new_all();
    let target_pid = args.pid.map(Pid::from_u32);

    if let Some(pid) = target_pid {
        println!("🎯 Monitoring process ID: {}", pid.as_u32());
    }

    let mut csv_writer = if let Some(export_path) = &args.export {
        let mut writer = csv::Writer::from_path(export_path)?;
        writer.write_record(&ResourceSnapshot::to_csv_header().split(',').collect::<Vec<_>>())?;
        Some(writer)
    } else {
        None
    };

    println!("\n🔄 Starting monitoring (interval: {} seconds)", args.interval);
    if args.continuous {
        println!("Press Ctrl+C to stop monitoring\n");
    }

    let start_time = Instant::now();
    let mut iteration = 0;

    loop {
        iteration += 1;
        let snapshot = ResourceSnapshot::new(&mut system, target_pid)?;
        
        println!("--- Iteration {} ({}s elapsed) ---", iteration, start_time.elapsed().as_secs());
        println!("{}", snapshot.format_summary());

        if let Some(ref mut writer) = csv_writer {
            writer.write_record(&snapshot.to_csv_row().split(',').collect::<Vec<_>>())?;
            writer.flush()?;
        }

        if !args.continuous {
            break;
        }

        thread::sleep(Duration::from_secs(args.interval));
    }

    if let Some(export_path) = &args.export {
        println!("📁 Monitoring data exported to: {}", export_path.display());
    }

    Ok(())
}
