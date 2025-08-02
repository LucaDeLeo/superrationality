import { describe, it, expect } from 'vitest'
import path from 'path'
import fs from 'fs'
import { parse } from 'jsonc-parser'

describe('Project Setup Validation', () => {
  it('should have required folder structure', () => {
    const requiredFolders = ['components', 'pages', 'hooks', 'services', 'types']
    const srcPath = path.resolve(__dirname, '..')

    requiredFolders.forEach((folder) => {
      const folderPath = path.join(srcPath, folder)
      expect(fs.existsSync(folderPath)).toBe(true)
    })
  })

  it('should have TypeScript configuration with strict mode', () => {
    const tsconfigPath = path.resolve(__dirname, '../../tsconfig.app.json')
    const tsconfigContent = fs.readFileSync(tsconfigPath, 'utf-8')
    const tsconfig = parse(tsconfigContent)
    
    expect(tsconfig.compilerOptions.strict).toBe(true)
    expect(tsconfig.compilerOptions.noUnusedLocals).toBe(true)
    expect(tsconfig.compilerOptions.noUnusedParameters).toBe(true)
  })

  it('should have path aliases configured', () => {
    const tsconfigPath = path.resolve(__dirname, '../../tsconfig.app.json')
    const tsconfigContent = fs.readFileSync(tsconfigPath, 'utf-8')
    const tsconfig = parse(tsconfigContent)
    
    expect(tsconfig.compilerOptions.paths).toBeDefined()
    expect(tsconfig.compilerOptions.paths['@/*']).toEqual(['src/*'])
    expect(tsconfig.compilerOptions.paths['@components/*']).toEqual(['src/components/*'])
    expect(tsconfig.compilerOptions.paths['@pages/*']).toEqual(['src/pages/*'])
    expect(tsconfig.compilerOptions.paths['@hooks/*']).toEqual(['src/hooks/*'])
    expect(tsconfig.compilerOptions.paths['@services/*']).toEqual(['src/services/*'])
    expect(tsconfig.compilerOptions.paths['@types/*']).toEqual(['src/types/*'])
  })

  it('should have ESLint configuration', () => {
    const eslintConfigPath = path.resolve(__dirname, '../../.eslintrc.cjs')
    expect(fs.existsSync(eslintConfigPath)).toBe(true)
  })

  it('should have Prettier configuration', () => {
    const prettierConfigPath = path.resolve(__dirname, '../../.prettierrc')
    expect(fs.existsSync(prettierConfigPath)).toBe(true)
    
    const prettierConfig = JSON.parse(fs.readFileSync(prettierConfigPath, 'utf-8'))
    expect(prettierConfig.semi).toBe(false)
    expect(prettierConfig.singleQuote).toBe(true)
    expect(prettierConfig.printWidth).toBe(100)
  })

  it('should have required npm scripts', () => {
    const packageJsonPath = path.resolve(__dirname, '../../package.json')
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'))
    
    const requiredScripts = ['dev', 'build', 'lint', 'lint:fix', 'format', 'test']
    requiredScripts.forEach((script) => {
      expect(packageJson.scripts[script]).toBeDefined()
    })
  })
})