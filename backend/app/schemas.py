from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ItemCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class ItemUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class ItemOut(BaseModel):
    id: int
    title: str
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class UserOut(BaseModel):
    id: int
    email: EmailStr
    model_config = ConfigDict(from_attributes=True)


class SignIn(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DiaryGenerateRequest(BaseModel):
    topic: str = Field("日常", min_length=1, max_length=80)
    max_retries: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.9, ge=0.0, le=2.0)


class DiaryGenerateResponse(BaseModel):
    sentence: str
    model: str
    retries_used: int
