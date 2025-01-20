#!/bin/bash


BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"

$BINARY_PATH symbol_conversion_normal.yaml
$BINARY_PATH symbol_conversion_preepileptic.yaml
$BINARY_PATH symbol_conversion_seizure.yaml
