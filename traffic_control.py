import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from enum import Enum
import threading


class TrafficLightState(Enum):
    """Traffic light states"""
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    FLASHING_RED = "flashing_red"


class TrafficIntersection:
    """Represents a traffic intersection with multiple lanes"""
    
    def __init__(self, intersection_id: str, lanes: List[str] = None):
        """
        Initialize a traffic intersection
        
        Args:
            intersection_id: Unique identifier for the intersection
            lanes: List of lane names (e.g., ['North', 'South', 'East', 'West'])
        """
        self.intersection_id = intersection_id
        self.lanes = lanes or ["North", "South", "East", "West"]
        self.traffic_lights = {lane: TrafficLightState.RED for lane in self.lanes}
        self.green_wave_active = False
        self.ambulance_detected = False
        self.ambulance_direction = None
        self.green_wave_start_time = None
        self.green_wave_duration = 30  # seconds
        
    def set_traffic_light(self, lane: str, state: TrafficLightState):
        """Set traffic light state for a specific lane"""
        if lane in self.traffic_lights:
            self.traffic_lights[lane] = state
            
    def get_traffic_light(self, lane: str) -> TrafficLightState:
        """Get current traffic light state for a lane"""
        return self.traffic_lights.get(lane, TrafficLightState.RED)
    
    def activate_green_wave(self, ambulance_direction: str):
        """
        Activate green wave for ambulance
        Sets ambulance direction to green and cross-traffic to red
        """
        self.ambulance_detected = True
        self.ambulance_direction = ambulance_direction
        self.green_wave_active = True
        self.green_wave_start_time = datetime.now()
        
        # Set ambulance direction to green
        self.set_traffic_light(ambulance_direction, TrafficLightState.GREEN)
        
        # Set cross-traffic to red
        for lane in self.lanes:
            if lane != ambulance_direction:
                self.set_traffic_light(lane, TrafficLightState.RED)
                
        return {
            "status": "green_wave_activated",
            "intersection_id": self.intersection_id,
            "ambulance_direction": ambulance_direction,
            "timestamp": datetime.now().isoformat()
        }
    
    def deactivate_green_wave(self):
        """Deactivate green wave and return to normal traffic flow"""
        self.green_wave_active = False
        self.ambulance_detected = False
        self.ambulance_direction = None
        self.green_wave_start_time = None
        
        # Reset to normal alternating pattern
        self._reset_normal_traffic()
        
        return {
            "status": "green_wave_deactivated",
            "intersection_id": self.intersection_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _reset_normal_traffic(self):
        """Reset to normal traffic light pattern"""
        # Simple alternating pattern: North-South green, then East-West green
        for lane in self.lanes:
            if lane in ["North", "South"]:
                self.set_traffic_light(lane, TrafficLightState.GREEN)
            else:
                self.set_traffic_light(lane, TrafficLightState.RED)
    
    def is_green_wave_expired(self) -> bool:
        """Check if green wave duration has expired"""
        if self.green_wave_start_time is None:
            return False
        
        elapsed = (datetime.now() - self.green_wave_start_time).total_seconds()
        return elapsed > self.green_wave_duration
    
    def get_status(self) -> Dict:
        """Get current intersection status"""
        return {
            "intersection_id": self.intersection_id,
            "traffic_lights": {lane: state.value for lane, state in self.traffic_lights.items()},
            "green_wave_active": self.green_wave_active,
            "ambulance_detected": self.ambulance_detected,
            "ambulance_direction": self.ambulance_direction,
            "timestamp": datetime.now().isoformat()
        }


class TrafficControlNetwork:
    """Manages a network of intersections for coordinated green wave"""
    
    def __init__(self):
        """Initialize traffic control network"""
        self.intersections: Dict[str, TrafficIntersection] = {}
        self.ambulance_route: List[str] = []
        self.ambulance_position = 0
        self.ambulance_speed = 5  # meters per second (estimate)
        self.ambulance_tracking = False
        self.update_lock = threading.Lock()
        
    def add_intersection(self, intersection_id: str, lanes: List[str] = None) -> TrafficIntersection:
        """Add a new intersection to the network"""
        intersection = TrafficIntersection(intersection_id, lanes)
        self.intersections[intersection_id] = intersection
        return intersection
    
    def set_ambulance_route(self, route: List[str]):
        """
        Set the ambulance route through intersections
        
        Args:
            route: List of intersection IDs in order
        """
        self.ambulance_route = route
        self.ambulance_position = 0
        self.ambulance_tracking = True
        
        return {
            "status": "route_set",
            "route": route,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_ambulance_position(self, current_intersection_id: str):
        """Update ambulance position and activate green wave"""
        with self.update_lock:
            if not self.ambulance_tracking or current_intersection_id not in self.intersections:
                return
            
            # Deactivate green wave at previous intersections
            for inter_id, intersection in self.intersections.items():
                if inter_id != current_intersection_id and intersection.green_wave_active:
                    intersection.deactivate_green_wave()
            
            # Activate green wave at current intersection
            current_intersection = self.intersections[current_intersection_id]
            
            # Determine ambulance direction (simplified: use predefined direction)
            ambulance_direction = self._get_ambulance_direction(current_intersection_id)
            
            return current_intersection.activate_green_wave(ambulance_direction)
    
    def _get_ambulance_direction(self, intersection_id: str) -> str:
        """
        Determine ambulance direction at intersection
        Simplified: returns primary direction
        """
        # In a real system, this would use GPS/traffic data
        return "North"  # Default direction
    
    def activate_green_wave_at_intersection(self, intersection_id: str, direction: str):
        """Manually activate green wave at specific intersection"""
        if intersection_id in self.intersections:
            return self.intersections[intersection_id].activate_green_wave(direction)
        return {"error": f"Intersection {intersection_id} not found"}
    
    def deactivate_green_wave_at_intersection(self, intersection_id: str):
        """Deactivate green wave at specific intersection"""
        if intersection_id in self.intersections:
            return self.intersections[intersection_id].deactivate_green_wave()
        return {"error": f"Intersection {intersection_id} not found"}
    
    def get_network_status(self) -> Dict:
        """Get status of entire traffic control network"""
        return {
            "timestamp": datetime.now().isoformat(),
            "ambulance_tracking": self.ambulance_tracking,
            "ambulance_route": self.ambulance_route,
            "intersections": {
                inter_id: inter.get_status() 
                for inter_id, inter in self.intersections.items()
            }
        }
    
    def stop_ambulance_tracking(self):
        """Stop tracking ambulance and reset network"""
        with self.update_lock:
            self.ambulance_tracking = False
            self.ambulance_route = []
            
            # Deactivate all green waves
            for intersection in self.intersections.values():
                if intersection.green_wave_active:
                    intersection.deactivate_green_wave()
            
            return {
                "status": "tracking_stopped",
                "timestamp": datetime.now().isoformat()
            }


# Global traffic control network instance
traffic_network = TrafficControlNetwork()

# Initialize sample intersections
def initialize_sample_network():
    """Initialize sample traffic control network with 4 roads in ring configuration (North, East, South, West)"""
    intersection_ids = ["NORTH", "EAST", "SOUTH", "WEST"]
    
    for inter_id in intersection_ids:
        traffic_network.add_intersection(inter_id)
    
    return intersection_ids
