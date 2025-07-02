"""
Global Resource Manager for handling shared resources across agents.

This module separates resource management from individual agent drives,
ensuring consistent resource regeneration in multi-agent environments.
"""

import numpy as np
from typing import Dict, Any, Optional
from .get_params import ParameterHandler


class GlobalResourceManager:
    """
    Manages shared resources across all agents in the environment.
    
    This class handles:
    - Resource regeneration based on global parameters
    - Consistent resource state across all agents
    - Centralized resource configuration
    """
    
    def __init__(self, config_path: str, drive_type: str):
        """
        Initialize the global resource manager.
        
        Args:
            config_path: Path to configuration file
            drive_type: Type of drive (used for configuration compatibility)
        """
        self.param_manager = ParameterHandler(config_path)
        
        # Get global resource configuration
        self.resource_config = self.param_manager.config['global_params']['optimal_internal_state']
        
        # Extract regeneration rates
        self.regeneration_rates = {
            state_name: state_config['regeneration'] 
            for state_name, state_config in self.resource_config.items()
        }
        
        # Get resource names for reference
        self.resource_names = list(self.resource_config.keys())
        
        # Random number generator for reproducible results
        self.rng = np.random.RandomState()
        
    def set_random_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducible resource regeneration."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
    def get_regeneration_rates(self) -> Dict[str, float]:
        """Get regeneration rates for all resource types."""
        return self.regeneration_rates.copy()
        
    def get_regeneration_rate(self, resource_name: str) -> float:
        """Get regeneration rate for a specific resource type."""
        if resource_name not in self.regeneration_rates:
            raise ValueError(f"Unknown resource type: {resource_name}")
        return self.regeneration_rates[resource_name]
        
    def get_resource_names(self) -> list:
        """Get list of all available resource names."""
        return self.resource_names.copy()
        
    def apply_resource_regeneration(self, resources_info: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Apply regeneration to all shared resources.
        
        Args:
            resources_info: Dictionary mapping resource_id -> resource_data
                          where resource_data contains 'name' and 'available' keys
                          
        Returns:
            Updated resources_info with regeneration applied
        """
        for resource_id, resource_data in resources_info.items():
            # Only attempt regeneration if resource is currently unavailable
            if not resource_data["available"]:
                state_name = resource_data["name"]
                
                if state_name in self.regeneration_rates:
                    regen_rate = self.regeneration_rates[state_name]
                    
                    # Single global decision for this resource
                    if self.rng.uniform(0, 1) < regen_rate:
                        resource_data["available"] = True
                        
        return resources_info
    
    def apply_resource_regeneration_single(self, resource_available: bool, resource_name: str) -> bool:
        """
        Apply regeneration to a single resource (backward compatibility).
        
        Args:
            resource_available: Current availability status
            resource_name: Name of the resource type
            
        Returns:
            Updated availability status
        """
        if resource_available:
            return True
            
        if resource_name not in self.regeneration_rates:
            raise ValueError(f"Unknown resource type: {resource_name}")
            
        regen_rate = self.regeneration_rates[resource_name]
        return self.rng.uniform(0, 1) < regen_rate
    
    def get_resource_stock_regeneration_array(self) -> np.ndarray:
        """
        Get regeneration rates as numpy array (for NORMARL-style environments).
        
        Returns:
            Array of regeneration rates in the order of resource_names
        """
        return np.array([
            self.regeneration_rates[name] for name in self.resource_names
        ], dtype=np.float32)
        
    def update_resource_stock(self, current_stock: np.ndarray, total_consumption: np.ndarray) -> np.ndarray:
        """
        Update global resource stock with regeneration and consumption.
        
        Args:
            current_stock: Current resource stock levels
            total_consumption: Total consumption across all agents
            
        Returns:
            Updated resource stock levels
        """
        regeneration_rates = self.get_resource_stock_regeneration_array()
        
        # Apply regeneration and subtract consumption
        new_stock = (1 + regeneration_rates) * current_stock - total_consumption
        
        # Ensure stock doesn't go below zero
        return np.maximum(0, new_stock)
        
    def get_resource_config(self) -> Dict[str, Dict[str, float]]:
        """Get the full resource configuration."""
        return self.resource_config.copy()
        
    def validate_resources_info(self, resources_info: Dict[int, Dict[str, Any]]) -> bool:
        """
        Validate that resources_info contains valid resource types.
        
        Args:
            resources_info: Resources information to validate
            
        Returns:
            True if all resources are valid, False otherwise
        """
        for resource_id, resource_data in resources_info.items():
            if "name" not in resource_data:
                return False
            if resource_data["name"] not in self.resource_names:
                return False
            if "available" not in resource_data:
                return False
                
        return True
        
    def get_stats(self, resources_info: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about current resource state.
        
        Args:
            resources_info: Current resources information
            
        Returns:
            Dictionary with resource statistics
        """
        stats = {}
        
        # Count available resources by type
        resource_counts = {}
        for resource_data in resources_info.values():
            name = resource_data["name"]
            available = resource_data["available"]
            
            if name not in resource_counts:
                resource_counts[name] = {"total": 0, "available": 0}
                
            resource_counts[name]["total"] += 1
            if available:
                resource_counts[name]["available"] += 1
                
        stats["resource_counts"] = resource_counts
        stats["total_resources"] = len(resources_info)
        stats["available_resources"] = sum(
            1 for r in resources_info.values() if r["available"]
        )
        stats["regeneration_rates"] = self.regeneration_rates.copy()
        
        return stats
        
    def __str__(self) -> str:
        """String representation of the resource manager."""
        available_resources = ", ".join(self.resource_names)
        return f"GlobalResourceManager(resources=[{available_resources}])" 
