import yaml
from drives.base_drive import BaseDrive
from drives.interoceptive_drive import InteroceptiveModulationDrive
from drives.elliptic_drive import EllipticDrive

class ParameterHandler():
	def __init__(self, parameter_path):
		with open(parameter_path, 'r') as file:
			self.config = yaml.safe_load(file)

	def create_drive(self, drive_type):
		base_params = self.config['drive_params']['base_drive'].copy()

		specific_params = self.config['drive_params'].get(drive_type, {})

		combined_params = {**base_params, **specific_params}

		optimal_internal_states_config = self.config['global_params']['optimal_internal_state']

		if drive_type == 'interoceptive_drive':
						return InteroceptiveModulationDrive(
								optimal_internal_states_config=optimal_internal_states_config,
								m=combined_params['m'],
								n=combined_params['n'],
								eta=combined_params['eta']
						)
		elif drive_type == "elliptic_drive":
						return EllipticDrive(
							optimal_internal_states_config=optimal_internal_states_config,
							n_vector=combined_params['n_vector'],
							m=combined_params['m']
						)
		else:
						return BaseDrive(
							optimal_internal_states_config=optimal_internal_states_config,
							m=combined_params['m'],
							n=combined_params['n']
						)

