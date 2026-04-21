#!/usr/bin/env python3
"""
NVIDIA Nsight Compute (ncu) 性能分析脚本
支持自动处理 sudo 权限
"""

import subprocess
import sys
import argparse
from pathlib import Path
import getpass
import shutil

def get_ncu_path():
    """获取 ncu 的完整路径"""
    ncu_path = shutil.which("ncu")
    if ncu_path:
        return ncu_path
    
    # 尝试常见的安装路径
    common_paths = [
        "/usr/local/cuda-13.0/bin/ncu",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None

def check_ncu_available():
    """检查 ncu 是否可用"""
    ncu_path = get_ncu_path()
    if ncu_path:
        print(f"找到 ncu: {ncu_path}")
        return True
    return False

def run_command_with_sudo(cmd, password):
    """使用 sudo 和密码运行命令"""
    # 使用 sudo -E 保留环境变量，或者直接用完整路径
    sudo_cmd = ["sudo", "-S"] + cmd
    
    try:
        process = subprocess.Popen(
            sudo_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 发送密码
        stdout, stderr = process.communicate(input=f"{password}\n")
        
        return process.returncode, stdout, stderr
    except Exception as e:
        return -1, "", str(e)

def run_ncu_profile(executable, kernel_name=None, metrics=None, output_file="profile_report", password=None):
    """执行 ncu 性能分析"""
    
    # 获取 ncu 的完整路径
    ncu_path = get_ncu_path()
    if not ncu_path:
        print("错误: 找不到 ncu 命令")
        return False
    
    # 构建 ncu 命令（使用完整路径）
    cmd = [
        ncu_path,
        "--export", output_file,
        "--force-overwrite",
        "--target-processes", "all"
    ]
    
    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])
    
    if metrics:
        cmd.extend(["--metrics", ",".join(metrics)])
    else:
        cmd.extend(["--set", "full"])
    
    cmd.append(str(Path(executable).resolve()))  # 使用绝对路径
    
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 80)
    
    # 第一次尝试：不用 sudo
    print("尝试直接运行 ncu...")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        # 打印输出
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        
        # 检查是否成功
        if process.returncode == 0:
            print("\n✓ 性能分析完成")
            return True
        
        # 检查是否是权限问题
        if "ERR_NVGPUCTRPERM" in stderr or "ERR_NVGPUCTRPERM" in stdout:
            print("\n检测到权限问题，需要使用 sudo...")
            
            # 如果没有提供密码，询问用户
            if password is None:
                password = getpass.getpass("请输入 sudo 密码: ")
            
            # 使用 sudo 重试
            print("\n使用 sudo 重新运行...")
            print(f"执行命令: sudo {' '.join(cmd)}")
            print("-" * 80)
            
            returncode, stdout, stderr = run_command_with_sudo(cmd, password)
            
            # 打印输出
            if stdout:
                print(stdout)
            if stderr:
                # 过滤掉 sudo 密码提示
                stderr_lines = [line for line in stderr.split('\n') 
                               if not line.startswith('[sudo]') and line.strip()]
                if stderr_lines:
                    print('\n'.join(stderr_lines), file=sys.stderr)
            
            if returncode == 0:
                print("\n✓ 性能分析完成（使用 sudo）")
                return True
            else:
                print(f"\n✗ 执行失败 (退出码: {returncode})")
                return False
        else:
            print(f"\n✗ 执行失败 (退出码: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_key_metrics(report_file, password=None):
    """从报告中提取关键性能指标"""
    report_path = f"{report_file}.ncu-rep"
    
    if not Path(report_path).exists():
        print(f"\n警告: 找不到报告文件 {report_path}")
        return
    
    print("\n" + "=" * 80)
    print("关键性能指标分析")
    print("=" * 80)
    
    ncu_path = get_ncu_path()
    cmd = [ncu_path, "--import", report_path, "--page", "summary"]
    
    try:
        # 先尝试不用 sudo
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            # 如果失败且有密码，尝试用 sudo
            if password:
                returncode, stdout, stderr = run_command_with_sudo(cmd, password)
                if returncode == 0:
                    print(stdout)
                else:
                    print("无法提取性能指标")
            else:
                print("无法提取性能指标")
                
    except subprocess.TimeoutExpired:
        print("提取性能指标超时")
    except Exception as e:
        print(f"提取性能指标时出错: {e}")

def print_optimization_tips():
    """打印优化建议"""
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    print("""
关键指标说明：
  • SM Throughput: 流多处理器吞吐率（目标 > 60%）
  • Memory Throughput: 内存带宽利用率
  • Warp Occupancy: Warp 占用率（目标 > 50%）
  • Register Usage: 寄存器使用情况
  • Shared Memory: 共享内存使用情况

常见优化方向：
  1. 如果 SM 吞吐率低 → 增加并行度（更多 blocks/threads）
  2. 如果内存带宽接近 100% → 优化内存访问模式（合并访问）
  3. 如果 Warp 占用率低 → 调整 block size 或减少寄存器使用
  4. 如果有大量 stall → 检查数据依赖和同步操作
  5. 使用 shared memory 减少全局内存访问
    """)

def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Nsight Compute 性能分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s ./my_cuda_app -k my_kernel -o my_report
  %(prog)s ./my_cuda_app --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed
        """
    )
    parser.add_argument("executable", help="要分析的 CUDA 可执行文件")
    parser.add_argument("-k", "--kernel", help="指定要分析的 kernel 名称（支持正则表达式）")
    parser.add_argument("-o", "--output", default="profile_report", help="输出报告文件名（默认: profile_report）")
    parser.add_argument("-m", "--metrics", nargs="+", help="自定义指标列表")
    parser.add_argument("-p", "--password", help="sudo 密码（不推荐，建议交互式输入）")
    
    args = parser.parse_args()
    
    # 检查 ncu 是否可用
    if not check_ncu_available():
        print("错误: 找不到 ncu 命令")
        print("请确保 NVIDIA Nsight Compute 已安装并添加到 PATH")
        print("\n安装方法:")
        print("  1. 从 NVIDIA 官网下载: https://developer.nvidia.com/nsight-compute")
        print("  2. 或使用 CUDA Toolkit 自带的 ncu")
        print("\n常见安装路径:")
        print("  /usr/local/cuda/bin/ncu")
        print("  /opt/nvidia/nsight-compute/ncu")
        sys.exit(1)
    
    # 检查可执行文件
    exe_path = Path(args.executable)
    if not exe_path.exists():
        print(f"错误: 找不到可执行文件 {args.executable}")
        sys.exit(1)
    
    if not exe_path.stat().st_mode & 0o111:
        print(f"错误: {args.executable} 没有执行权限")
        print(f"请运行: chmod +x {args.executable}")
        sys.exit(1)
    
    # 开始性能分析
    print("=" * 80)
    print("NVIDIA Nsight Compute 性能分析")
    print("=" * 80)
    print(f"目标程序: {args.executable}")
    if args.kernel:
        print(f"目标 Kernel: {args.kernel}")
    print(f"输出文件: {args.output}.ncu-rep")
    print("=" * 80)
    print()
    
    # 执行分析
    success = run_ncu_profile(
        args.executable, 
        args.kernel, 
        args.metrics, 
        args.output,
        args.password
    )
    
    if not success:
        print("\n性能分析失败")
        print("\n永久解决权限问题的方法:")
        print("sudo bash -c 'echo \"options nvidia NVreg_RestrictProfilingToAdminUsers=0\" > /etc/modprobe.d/nvidia-profiling.conf'")
        print("sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null; sudo modprobe nvidia")
        print("或重启系统")
        sys.exit(1)
    
    # 提取关键指标
    extract_key_metrics(args.output, args.password)
    
    # 打印优化建议
    print_optimization_tips()
    
    # 总结
    print("=" * 80)
    print("分析完成")
    print("=" * 80)
    print(f"报告文件: {args.output}.ncu-rep")
    print(f"查看详细报告: ncu-ui {args.output}.ncu-rep")
    print("=" * 80)

if __name__ == "__main__":
    main()
