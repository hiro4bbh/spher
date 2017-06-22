// Package spher provides sparse matrix functionalities.
package spher

import "runtime"

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
}
