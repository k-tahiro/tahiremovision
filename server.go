package main

import (
	"github.com/labstack/echo"
	echoMw "github.com/labstack/echo/middleware"

	"./handler"
	myMw "./middleware"
)

func main() {
	// Echo instance
	e := echo.New()

	// Middleware
	e.Use(echoMw.Logger())
	e.Use(echoMw.Recover())

	// カスタムコンテキスト用Middlewareを適用
	e.Use(myMw.MyCustomContextMiddleware())

	// DB用Middlewareを適用
	e.Use(myMw.SQLiteMiddleware("./command.db"))

	// Routes
	e.GET("/", handler.Hello)
	// Routes
	commands := e.Group("/commands")
	{
		commands.GET("/", handler.Commands)
		commands.POST("/transmit/:id", handler.Transmit)
		commands.POST("/receive", handler.Receive)
	}

	// Start server
	e.Logger.Fatal(e.Start(":1323"))
}
