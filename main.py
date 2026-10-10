import math
import random
import time
import statistics
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PolarPoint:
    radius: float
    angle: float

    @staticmethod
    def from_cartesian(p: 'CartesianPoint2D') -> 'PolarPoint':
        r = math.hypot(p.x, p.y)
        angle = math.atan2(p.y, p.x)
        return PolarPoint(r, angle)

@dataclass(frozen=True, slots=True)
class CartesianPoint2D:
    x: float
    y: float

    @staticmethod
    def from_polar(p: PolarPoint) -> 'CartesianPoint2D':
        x = p.radius * math.cos(p.angle)
        y = p.radius * math.sin(p.angle)
        return CartesianPoint2D(x, y)

@dataclass(frozen=True, slots=True)
class SphericalPoint:
    radius: float
    azimuth: float
    polarAngle: float

    @staticmethod
    def from_cartesian(p: 'CartesianPoint3D') -> 'SphericalPoint':
        r = math.hypot(p.x, p.y, p.z)
        azimuth = math.atan2(p.y, p.x)
        polarAngle = math.acos(p.z / r) if r != 0 else 0
        return SphericalPoint(r, azimuth, polarAngle)

@dataclass(frozen=True, slots=True)
class CartesianPoint3D:
    x: float
    y: float
    z: float

    @staticmethod
    def from_spherical(p: SphericalPoint) -> 'CartesianPoint3D':
        x = p.radius * math.sin(p.polarAngle) * math.cos(p.azimuth)
        y = p.radius * math.sin(p.polarAngle) * math.sin(p.azimuth)
        z = p.radius * math.cos(p.polarAngle)
        return CartesianPoint3D(x, y, z)

def distance_2d_cartesian(p1: CartesianPoint2D, p2: CartesianPoint2D) -> float:
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def distance_2d_polar(p1: PolarPoint, p2: PolarPoint) -> float:
    return math.sqrt(p1.radius**2 + p2.radius**2 - 2 * p1.radius * p2.radius * math.cos(p2.angle - p1.angle))

def _spherical_angle_term(p1: SphericalPoint, p2: SphericalPoint) -> float:
    t = (math.sin(p1.polarAngle) * math.sin(p2.polarAngle) * math.cos(p1.azimuth - p2.azimuth) +
         math.cos(p1.polarAngle) * math.cos(p2.polarAngle))
    return max(-1.0, min(1.0, t))

def distance_3d_spherical_chord(p1: SphericalPoint, p2: SphericalPoint) -> float:
    t = _spherical_angle_term(p1, p2)
    return math.sqrt(abs(p1.radius**2 + p2.radius**2 - 2 * p1.radius * p2.radius * t))

def distance_3d_spherical_arc(p1: SphericalPoint, p2: SphericalPoint) -> float:
    t = _spherical_angle_term(p1, p2)
    return p1.radius * math.acos(t)

def distance_3d_cartesian(p1: CartesianPoint3D, p2: CartesianPoint3D) -> float:
    return math.dist((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))

def test_correctness():
    print("--- Перевірка перетворень ---")
    p2_orig = CartesianPoint2D(3.0, 4.0)
    polar = PolarPoint.from_cartesian(p2_orig)
    p2_restored = CartesianPoint2D.from_polar(polar)
    print(f"2D Початкова: {p2_orig} -> Відновлена: {p2_restored}")

    p3_orig = CartesianPoint3D(1.0, 2.0, 3.0)
    spherical = SphericalPoint.from_cartesian(p3_orig)
    p3_restored = CartesianPoint3D.from_spherical(spherical)
    print(f"3D Початкова: {p3_orig} -> Відновлена: {p3_restored}\n")

def run_benchmarks():
    N = 100_000
    print(f"Генерація даних ({N} пар)...")
    
    polar_pairs = [(PolarPoint(random.uniform(0, 100), random.uniform(-math.pi, math.pi)),
                    PolarPoint(random.uniform(0, 100), random.uniform(-math.pi, math.pi)))
                   for _ in range(N)]
    cart_2d_pairs = [(CartesianPoint2D.from_polar(p1), CartesianPoint2D.from_polar(p2)) 
                     for p1, p2 in polar_pairs]
    
    spherical_pairs = []
    for _ in range(N):
        r = random.uniform(0.1, 100)
        spherical_pairs.append((
            SphericalPoint(r, random.uniform(-math.pi, math.pi), random.uniform(0, math.pi)),
            SphericalPoint(r, random.uniform(-math.pi, math.pi), random.uniform(0, math.pi))
        ))
    
    cart_3d_pairs = [(CartesianPoint3D.from_spherical(p1), CartesianPoint3D.from_spherical(p2))
                     for p1, p2 in spherical_pairs]
                     
    RUNS = 5
    
    def measure(func, data_pairs):
        times = []
        for _ in range(RUNS):
            start = time.perf_counter()
            for p1, p2 in data_pairs:
                func(p1, p2)
            times.append(time.perf_counter() - start)
        return statistics.median(times)
        
    print(f"Запуск бенчмарків (Медіана з {RUNS} прогонів)...")
    
    t_polar_2d = measure(distance_2d_polar, polar_pairs)
    t_cart_2d = measure(distance_2d_cartesian, cart_2d_pairs)
    
    t_sph_chord = measure(distance_3d_spherical_chord, spherical_pairs)
    t_sph_arc = measure(distance_3d_spherical_arc, spherical_pairs)
    t_cart_3d = measure(distance_3d_cartesian, cart_3d_pairs)
    
    print("\n--- Результати (Медіанний час у секундах) ---")
    print(f"2D Полярна відстань:     {t_polar_2d:.5f} с")
    print(f"2D Декартова відстань:   {t_cart_2d:.5f} с")
    print(f"3D Сферична (хорда):     {t_sph_chord:.5f} с")
    print(f"3D Сферична (дуга):      {t_sph_arc:.5f} с")
    print(f"3D Декартова відстань:   {t_cart_3d:.5f} с")

if __name__ == '__main__':
    test_correctness()
    run_benchmarks()