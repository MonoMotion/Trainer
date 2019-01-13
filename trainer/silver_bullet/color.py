import dataclasses

@dataclasses.dataclass
class Color:
    r: float
    g: float
    b: float
    a: float = 1

    def as_rgb(self):
        return self.as_rgba()[:3]

    def as_rgba(self):
        return [self.r, self.g, self.b, self.a]

    @staticmethod
    def red():
        return Color(1, 0, 0, 1)

    @staticmethod
    def green():
        return Color(0, 1, 0, 1)

    @staticmethod
    def blue():
        return Color(0, 0, 1, 1)

