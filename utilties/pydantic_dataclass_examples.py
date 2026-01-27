from pydantic import BaseModel
from datetime import datetime
from dataclasses import dataclass


class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None = None

    def __str__(self):
        return str(self.id) + " " + self.name


# Validation happens automatically on instantiation


@dataclass
class DataUser:
    id: int
    name: str = "John Doe"

    def __str__(self):
        return str(self.id) + " " + self.name


def main():
    user = User(id=123)
    data_user = DataUser(id=123)
    print(user)
    print(data_user)


if __name__ == "__main__":
    main()
