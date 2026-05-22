# -*- coding: utf-8 -*-
"""
CARLA Automatic Emergency Braking (AEB) & Radar Collision Test Script
This script implements a complete, visual simulation of an Ego vehicle testing
automatic emergency braking against a static obstacle vehicle, using CARLA's
Traffic Manager. It features a side-by-side camera GUI (RGB + Semantic), 
a structured HUD, front bumper radar detection, video recording, 
performance plotting, and automatic termination upon a safe stop.
"""

import time
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    import carla
except ImportError:
    print("【Import Error】CARLA library not found. Please check your Python path.")
    sys.exit(1)

# Import our custom Sensor Manager
sys.path.append(os.path.dirname(__file__))
from carla_sensor_manager import CarlaSensorManager

def main():
    print("====================================================")
    print(" CARLA AEB & Radar Collision Avoidance Visual Test")
    print("====================================================")
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("Connected to CARLA Server.")
    except Exception as e:
        print(f"【Connection Error】Could not connect to CARLA Server: {e}")
        sys.exit(1)
        
    # Synchronous mode settings (20 FPS)
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    # Initialize Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    
    # Actors cleanup list
    actor_list = []
    sensor_manager = None
    video_writer = None
    
    # Telemetry logging lists
    log_time = []
    log_speed = []
    log_actual_dist = []
    log_radar_dist = []
    log_radar_vel = []
    
    # Sensor data variables
    latest_lidar_points = None
    latest_radar_points = None
    
    try:
        blueprint_library = world.get_blueprint_library()
        
        # 1. Spawn Ego Vehicle (Tesla Model 3)
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        # Select first spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        
        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Ego Vehicle (Model 3) spawned at: {spawn_point.location}. ID: {ego_vehicle.id}")
        
        # 2. Spawn Static Obstacle Vehicle (Tesla Model 3) 10m ahead
        obstacle_bp = blueprint_library.find('vehicle.tesla.model3')
        # Select different color for obstacle to easily distinguish
        obstacle_bp.set_attribute('color', '255,0,0') # Red
        
        forward_vector = spawn_point.get_forward_vector()
        obstacle_vehicle = None
        
        # Retry loop to spawn obstacle safely without collision errors
        for attempt in range(3):
            spawn_dist = 10.0 + attempt * 2.0
            obs_location = spawn_point.location + forward_vector * spawn_dist
            obs_location.z += 0.3  # Offset Z slightly to avoid ground collision
            obstacle_transform = carla.Transform(obs_location, spawn_point.rotation)
            
            obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_transform)
            if obstacle_vehicle:
                actor_list.append(obstacle_vehicle)
                print(f"Obstacle Vehicle (Red Model 3) spawned {spawn_dist}m ahead. ID: {obstacle_vehicle.id}")
                obstacle_vehicle.set_autopilot(False)
                break
                
        if not obstacle_vehicle:
            print("【Error】Could not spawn obstacle vehicle. Aborting test.")
            return
            
        # 3. Setup Multi-Sensor Suite via CarlaSensorManager
        sensor_manager = CarlaSensorManager(world, ego_vehicle)
        
        # Camera transforms (windshield mount, looking ahead)
        cam_location = carla.Location(x=1.6, y=0.0, z=1.2)
        cam_rotation = carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        
        sensor_manager.spawn_rgb_camera(cam_transform, role_name='rgb_front')
        sensor_manager.spawn_semantic_segmentation_camera(cam_transform, role_name='seg_front')
        
        # LiDAR transform (roof mount)
        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4))
        sensor_manager.spawn_lidar(lidar_transform, role_name='lidar')
        
        # Radar transform (front bumper)
        radar_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=0.4))
        sensor_manager.spawn_radar(radar_transform, role_name='radar')
        
        # IMU and GNSS
        sensor_manager.spawn_imu(carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)), role_name='imu')
        sensor_manager.spawn_gnss(carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4)), role_name='gnss')
        
        print("Sensors spawned and attached.")
        
        # 4. Phase 1: Physical Settling Phase (1.0 second / 20 steps)
        # Apply handbrake to Ego and tick to let both vehicles settle on the road mesh naturally.
        print("Settle Phase: Letting vehicles settle on the road surface...")
        for _ in range(20):
            ego_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
            world.tick()
            
        # Freeze obstacle vehicle's physics to keep it completely static
        obstacle_vehicle.set_simulate_physics(False)
        print("Obstacle vehicle physics frozen on the road surface.")
        
        # 5. Enable Autopilot (Traffic Manager) on Ego Vehicle
        ego_vehicle.set_autopilot(True)
        traffic_manager.set_global_distance_to_leading_vehicle(4.0) # Set safety gap to 4.0m
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0)
        traffic_manager.ignore_signs_percentage(ego_vehicle, 100.0)
        
        print("\nStarting Test Loop. Autopilot active. Watch the AEB system work.")
        print("Press Ctrl+C to abort manually.\n")
        
        step_count = 0
        consecutive_stopped = 0
        
        while True:
            # Step the simulation
            world.tick()
            step_count += 1
            t_sim = step_count * 0.05
            
            # Fetch physical states
            ego_loc = ego_vehicle.get_transform().location
            obs_loc = obstacle_vehicle.get_transform().location
            actual_distance = ego_loc.distance(obs_loc)
            
            velocity = ego_vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Read sensor data
            img_rgb = sensor_manager.get_image('rgb_front')
            img_seg = sensor_manager.get_image('seg_front')
            lidar_data = sensor_manager.get_sensor_data('lidar')
            radar_data = sensor_manager.get_sensor_data('radar')
            imu_data = sensor_manager.get_sensor_data('imu')
            gnss_data = sensor_manager.get_sensor_data('gnss')
            
            if lidar_data is not None:
                latest_lidar_points = lidar_data
            if radar_data is not None:
                latest_radar_points = radar_data
                
            # Process radar target tracking
            radar_closest_dist = 999.0
            radar_closest_vel = 0.0
            if latest_radar_points is not None and len(latest_radar_points) > 0:
                closest_idx = np.argmin(latest_radar_points[:, 3])
                radar_closest_dist = latest_radar_points[closest_idx, 3]
                radar_closest_vel = latest_radar_points[closest_idx, 0]
                
            # Log telemetry
            log_time.append(t_sim)
            log_speed.append(speed_kmh)
            log_actual_dist.append(actual_distance)
            log_radar_dist.append(radar_closest_dist if radar_closest_dist < 100.0 else 0.0)
            log_radar_vel.append(radar_closest_vel if radar_closest_dist < 100.0 else 0.0)
            
            # Console output
            radar_log_status = f"Dist: {radar_closest_dist:.2f} m, RelV: {radar_closest_vel:.2f} m/s" if radar_closest_dist < 100.0 else "No Target"
            print(f"Step {step_count:03d} | Speed: {speed_kmh:4.1f} km/h | Dist: {actual_distance:5.2f} m | Radar: [{radar_log_status}]")
            
            # 6. Check automatic stop and collision conditions
            if actual_distance < 10.0 and speed_kmh < 0.1:
                consecutive_stopped += 1
                if consecutive_stopped >= 15:
                    print("\n[SUCCESS] Ego vehicle successfully stopped in front of the obstacle!")
                    print(f"Stopped safely. Final Distance to obstacle: {actual_distance:.2f} m")
                    break
            else:
                consecutive_stopped = 0
                
            if actual_distance < 3.8:
                print("\n[COLLISION] Vehicles collided or got too close!")
                break
                
            if step_count > 400: # Max 20 seconds
                print("\n[TIMEOUT] Test reached maximum duration.")
                break
                
            # 7. Render OpenCV Dashboard (Compact 960x420 layout)
            if img_rgb is not None and img_seg is not None:
                img_resized = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2BGR), (480, 270))
                seg_resized = cv2.resize(cv2.cvtColor(img_seg, cv2.COLOR_BGRA2BGR), (480, 270))
                
                # Text labels on cameras
                cv2.putText(img_resized, "EGO VEHICLE RGB VIEW", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(seg_resized, "SEMANTIC SEGMENTATION VIEW", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Stack cameras side-by-side
                video_area = np.hstack((img_resized, seg_resized))
                
                # Create HUD black strip (960x150)
                hud_area = np.zeros((150, 960, 3), dtype=np.uint8)
                cv2.line(hud_area, (0, 0), (960, 0), (80, 80, 80), 2)
                cv2.line(hud_area, (310, 0), (310, 150), (80, 80, 80), 1)
                cv2.line(hud_area, (610, 0), (610, 150), (80, 80, 80), 1)
                
                dashboard = np.vstack((video_area, hud_area))
                
                # Column 1: Ego Telemetry
                cv2.putText(dashboard, "1. Ego Vehicle Telemetry", (15, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Speed: {speed_kmh:.1f} km/h", (15, 315), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Autopilot: ACTIVE (TM)", (15, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Ego State: Braking/Cruising", (15, 365), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Column 2: Physical & Radar Distance
                cv2.putText(dashboard, "2. Collision Telemetry", (325, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Actual Distance: {actual_distance:.2f} m", (325, 315), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Radar Distance: {radar_closest_dist:.2f} m" if radar_closest_dist < 100.0 else "Radar Distance: N/A", (325, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Relative Speed: {radar_closest_vel:.2f} m/s", (325, 365), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Column 3: Multi-Sensor Perception
                cv2.putText(dashboard, "3. Multi-Sensor Perception", (625, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                
                lidar_count = len(lidar_data) if lidar_data is not None else 0
                radar_count = len(radar_data) if radar_data is not None else 0
                cv2.putText(dashboard, f"LiDAR: {lidar_count} pts | Radar: {radar_count} dets", (625, 315), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                if imu_data is not None and gnss_data is not None:
                    ax, ay, az = imu_data['accel']
                    cv2.putText(dashboard, f"IMU Accel: [{ax:.1f}, {ay:.1f}, {az:.1f}] | GPS Lat: {gnss_data['lat']:.5f}", (625, 340), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(dashboard, f"Compass: {math.degrees(imu_data['compass']):.1f} deg | GPS Lon: {gnss_data['lon']:.5f}", (625, 365), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Collision warnings
                if actual_distance < 15.0:
                    cv2.putText(dashboard, f"!!! COLLISION WARNING: {actual_distance:.1f}m !!!", (300, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Show GUI Window
                cv2.imshow("CARLA AEB Collision Test Dashboard", dashboard)
                cv2.waitKey(1)
                
                # Setup VideoWriter
                if video_writer is None:
                    h, w, _ = dashboard.shape
                    out_path = os.path.join(os.path.dirname(__file__), "carla_collision_test_recording.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                    print(f"Video recording started: {out_path}")
                    
                if video_writer is not None:
                    video_writer.write(dashboard)
                    
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nTest stopped manually by user.")
    finally:
        # Stop video writer
        if video_writer is not None:
            video_writer.release()
            print("Video recording saved successfully.")
            
        # Clean up sensors and actors
        print("\nCleaning up spawned actors...")
        if sensor_manager:
            try:
                sensor_manager.destroy()
                print("Sensors cleaned up.")
            except Exception as e:
                print(f"Error cleaning up sensors: {e}")
                
        for actor in reversed(actor_list):
            if actor.is_alive:
                try:
                    actor.destroy()
                    print(f"Destroyed actor ID: {actor.id}")
                except Exception as e:
                    print(f"Failed to destroy actor ID {actor.id}: {e}")
                    
        # Restore sync settings
        try:
            world.apply_settings(original_settings)
            print("CARLA synchronous mode restored to original.")
        except Exception:
            pass
            
        cv2.destroyAllWindows()
        
        # 8. Generate and save performance graphs
        if len(log_time) > 5:
            print("\nGenerating and saving performance telemetry graphs...")
            
            # Graph 1: Speed & Distances
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            ax1.plot(log_time, log_speed, 'b-', label='Ego Speed (km/h)', linewidth=2)
            ax1.set_title("Ego Vehicle Speed during Emergency Braking (AEB)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Speed (km/h)")
            ax1.grid(True)
            ax1.legend(loc='upper right')
            
            ax2.plot(log_time, log_actual_dist, 'r-', label='Actual Distance (m)', linewidth=2)
            ax2.plot(log_time, log_radar_dist, 'g--', label='Radar Measured Distance (m)', linewidth=1.5)
            ax2.set_title("Distance to Obstacle vs Time")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Distance (meters)")
            ax2.grid(True)
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plot_path = os.path.join(os.path.dirname(__file__), "carla_collision_test_telemetry.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Telemetry graph saved: {plot_path}")
            
            # Graph 2: Radar target details
            fig2, ax_vel = plt.subplots(figsize=(10, 4))
            ax_vel.plot(log_time, log_radar_vel, 'm-', label='Radar Relative Speed (m/s)', linewidth=2)
            ax_vel.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax_vel.set_title("Radar Measured Relative Speed of Obstacle")
            ax_vel.set_xlabel("Time (s)")
            ax_vel.set_ylabel("Relative Velocity (m/s)")
            ax_vel.grid(True)
            ax_vel.legend()
            
            plot_path_radar = os.path.join(os.path.dirname(__file__), "carla_collision_test_radar.png")
            plt.savefig(plot_path_radar, dpi=150)
            plt.close()
            print(f"Radar tracking graph saved: {plot_path_radar}")
            
        print("Test sequence finished.")

if __name__ == '__main__':
    main()
