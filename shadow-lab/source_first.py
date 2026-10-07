"""Preview compatibility imports; production and previews share one implementation."""
from vehicle_pipeline.shadow_selection import (
    expand_ground_shadow, deepen_weak_footprint, choose_shadow,
    finish_ground_shadow, shadow_from_photo,
)
