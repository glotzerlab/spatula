 # Defaults to site-packages so only need package name
 install(DIRECTORY ./ DESTINATION spatula FILES_MATCHING PATTERN "*.py")
 install(FILES sphere-codes.npz DESTINATION spatula)
